import torch

from utils.util import opt_get


def create_batch_size_optimizer(opt_train):
    if 'batch_size_optimizer' in opt_train.keys():
        if opt_train['batch_size_optimizer']['type'] == 'gradient_direction':
            return GradientDirectionOptimizer(opt_train)
    return MegabatchBatchSizeOptimizer(opt_train)


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
class GradientDirectionOptimizer(BatchSizeOptimizer):
    def __init__(self, opt_train):
        self.mbf = opt_train['mega_batch_factor']
        self.opt = opt_train['batch_size_optimizer']
        self.max_full_batches = opt_get(self.opt, ['max_full_batches'], 10)
        self.parameters_to_poll = opt_get(self.opt, ['poll_parameters'], 8)
        self.recalculate_directions_every = opt_get(self.opt, ['recalculate_directions_steps'], 1)
        self.last_number_iterations = 0

    def vector_angle(self, v1, v2):
        with torch.no_grad():
            v1 = v1.flatten()
            v2 = v2.flatten()
            v1_norm = (v1 ** 2).sum().sqrt()
            v2_norm = (v2 ** 2).sum().sqrt()
            angle = torch.arccos((v1 * v2) / (v1_norm * v2_norm))
            return angle

    def focus(self, optimizer):
        optimizer._gradient_direction_optimizer_params = []
        optimizer._gradient_direction_optimizer_prior_directions = []
        optimizer._gradient_direction_optimizer_prior_grads = []
        optimizer._gradient_direction_optimizer_direction_change_magnitudes = []
        optimizer._gradient_direction_optimizer_step = 0
        self.current_opt = optimizer

    def should_step(self, it):
        self.last_number_iterations += 1

    def get_statistics(self):
        return {"last_number_iterations_before_step": self.last_number_iterations}
