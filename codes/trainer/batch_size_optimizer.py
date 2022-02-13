import math
import random

import torch
from torch import distributed
from torch._C._distributed_c10d import ReduceOp

from utils.util import opt_get


def create_batch_size_optimizer(opt_train):
    if 'batch_size_optimizer' in opt_train.keys():
        if opt_train['batch_size_optimizer']['type'] == 'gradient_direction':
            return GradientDirectionOptimizer(opt_train)
    return MegabatchBatchSizeOptimizer(opt_train)


def grad(p):
    if p.grad is None:
        return torch.tensor(0)
    return p.grad.detach().clone()


# Base class for BatchSizeOptimizers.
class BatchSizeOptimizer:
    def focus(self, optimizer):
        pass

    def should_step(self, it):
        raise NotImplementedError

    def get_statistics(self):
        return {}


# BatchSizeOptimizer that just steps every megabatch.
class MegabatchBatchSizeOptimizer(BatchSizeOptimizer):
    def __init__(self, opt_train):
        pass

    def should_step(self, it):
        return True


# BatchSizeOptimizer that uses the gradient direction of a few parameters to determine when to step.
# Very similar to what is described in https://aclanthology.org/2020.acl-main.323.pdf
# Special note: this class will ALWAYS accumulate, at a minimum, 3 batches. Plan accordingly.
class GradientDirectionOptimizer(BatchSizeOptimizer):
    def __init__(self, opt_train):
        self.opt = opt_train['batch_size_optimizer']
        self.max_full_batches = opt_get(self.opt, ['max_full_batches'], 10)
        self.parameters_to_poll = opt_get(self.opt, ['poll_parameters'], 8)
        self.recalculate_directions_every = opt_get(self.opt, ['recalculate_directions_steps'], 1)
        self.current_model = None

        # Metrics
        self.steps_taken = 0
        self.last_number_iterations = torch.zeros((128,))
        self.last_number_iterations_i = 0
        self.last_number_iterations_filled = False

    def vector_angle(self, v1, v2):
        if torch.all(v1 == 0) or torch.all(v2 == 0):
            return torch.tensor(0, device=v1.device)
        with torch.no_grad():
            v1 = v1.flatten()
            v2 = v2.flatten()
            v1_norm = (v1 ** 2).sum().sqrt()
            v2_norm = (v2 ** 2).sum().sqrt()
            angle = torch.arccos((torch.dot(v1, v2)) / (v1_norm * v2_norm))
            return angle

    def focus(self, model):
        if not hasattr(model, '_gradient_direction_optimizer_finished') or model._gradient_direction_optimizer_finished:
            all_params = list(filter(lambda t: '.weight' in t[0] and not hasattr(t[1].requires_grad, 'DO_NOT_TRAIN'),
                                     list(model.named_parameters())))  # Extracts weight parameters. Who cares about biases anyways? :)
            num_params = min(len(all_params), self.parameters_to_poll)
            model._gradient_direction_optimizer_params = random.sample(all_params, num_params)
            model._gradient_direction_optimizer_prior_directions = [0 for _ in range(num_params)]
            model._gradient_direction_optimizer_stopped_decreasing = [False for _ in range(num_params)]
            model._gradient_direction_optimizer_prior_grads = None
            model._gradient_direction_optimizer_step = 0
            model._gradient_direction_optimizer_finished = False
        self.current_model = model

    def should_step(self, it):
        model = self.current_model
        model._gradient_direction_optimizer_step += 1
        cur_grads = [grad(p) for k, p in model._gradient_direction_optimizer_params]
        for cg in cur_grads:
            if torch.any(torch.isnan(cg)):
                print("BSO: found NaN. Passing it off to the GradScaler..")
                return True
        if model._gradient_direction_optimizer_prior_grads is not None:
            cur_dir = [self.vector_angle(lgrad, cgrad) for lgrad, cgrad in zip(model._gradient_direction_optimizer_prior_grads, cur_grads)]
            delta_dir = [(cdir - ldir) for cdir, ldir in zip(cur_dir, model._gradient_direction_optimizer_prior_directions)]
            model._gradient_direction_optimizer_prior_directions = cur_dir
            model._gradient_direction_optimizer_stopped_decreasing = [sd or dd < 0 for sd, dd in zip(model._gradient_direction_optimizer_stopped_decreasing, delta_dir)]
            all_finished = all(model._gradient_direction_optimizer_stopped_decreasing)

            # For distributed optimizers, like ZeroRedundancyAdam, we need to reach a consensus as to whether or not to reduce.
            if distributed.is_initialized() and distributed.get_world_size() > 1:
                all_finished = torch.tensor(all_finished)
                distributed.all_reduce(all_finished, ReduceOp.BAND)
                all_finished = torch.all(all_finished)

            if all_finished or model._gradient_direction_optimizer_step >= self.max_full_batches:
                # <0 means the gradient direction is getting larger. Halt batch accumulation here.
                model._gradient_direction_optimizer_finished = True
                self.record_number_steps(model._gradient_direction_optimizer_step)
                # Fix the gradients. We've accumulated _gradient_direction_optimizer_step steps total, so we need to divide the grads by that.
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad = p.grad / model._gradient_direction_optimizer_step
                return True
        model._gradient_direction_optimizer_prior_grads = cur_grads
        return False

    def record_number_steps(self, steps):
        self.last_number_iterations[self.last_number_iterations_i] = steps
        if self.last_number_iterations_i == self.last_number_iterations.shape[0]-1:
            self.last_number_iterations_filled = True
        self.last_number_iterations_i = (self.last_number_iterations_i + 1) % self.last_number_iterations.shape[0]
        self.steps_taken += 1

    def get_statistics(self):
        res = {"batch_size_opt_total_steps": self.steps_taken}
        if self.last_number_iterations_filled:
            res["batch_size_opt_avg_iterations_per_step"] = self.last_number_iterations.mean().item()
        else:
            res["batch_size_opt_avg_iterations_per_step"] = self.last_number_iterations[:self.last_number_iterations_i].mean().item()
        return res
