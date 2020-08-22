from utils.loss_accumulator import LossAccumulator
from torch.nn import Module
import logging
from models.steps.losses import create_generator_loss
import torch
from apex import amp
from collections import OrderedDict
from .injectors import create_injector

logger = logging.getLogger('base')


# Defines the expected API for a single training step
class ConfigurableStep(Module):

    def __init__(self, opt_step, env):
        super(ConfigurableStep, self).__init__()

        self.step_opt = opt_step
        self.env = env
        self.opt = env['opt']
        self.gen = env['generators'][opt_step['generator']]
        self.discs = env['discriminators']
        self.gen_outputs = opt_step['generator_outputs']
        self.training_net = env['generators'][opt_step['training']] if opt_step['training'] in env['generators'].keys() else env['discriminators'][opt_step['training']]
        self.loss_accumulator = LossAccumulator()

        self.injectors = []
        if 'injectors' in self.step_opt.keys():
            for inj_name, injector in self.step_opt['injectors'].items():
                self.injectors.append(create_injector(injector, env))

        losses = []
        self.weights = {}
        for loss_name, loss in self.step_opt['losses'].items():
            losses.append((loss_name, create_generator_loss(loss, env)))
            self.weights[loss_name] = loss['weight']
        self.losses = OrderedDict(losses)

        # Intentionally abstract so subclasses can have alternative optimizers.
        self.define_optimizers()

    # Subclasses should override this to define individual optimizers. They should all go into self.optimizers.
    #  This default implementation defines a single optimizer for all Generator parameters.
    def define_optimizers(self):
        optim_params = []
        for k, v in self.training_net.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.env['rank'] <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        opt = torch.optim.Adam(optim_params, lr=self.step_opt['lr'],
                               weight_decay=self.step_opt['weight_decay'],
                               betas=(self.step_opt['beta1'], self.step_opt['beta2']))
        self.optimizers = [opt]

    # Returns all optimizers used in this step.
    def get_optimizers(self):
        assert self.optimizers is not None
        return self.optimizers

    # Returns optimizers which are opting in for default LR scheduling.
    def get_optimizers_with_default_scheduler(self):
        assert self.optimizers is not None
        return self.optimizers

    # Returns the names of the networks this step will train. Other networks will be frozen.
    def get_networks_trained(self):
        return [self.step_opt['training']]

    # Performs all forward and backward passes for this step given an input state. All input states are lists of
    # chunked tensors. Use grad_accum_step to dereference these steps. Should return a dict of tensors that later
    # steps might use. These tensors are automatically detached and accumulated into chunks.
    def do_forward_backward(self, state, grad_accum_step, amp_loss_id, backward=True):
        # First, do a forward pass with the generator.
        results = self.gen(state[self.step_opt['generator_input']][grad_accum_step])
        # Extract the resultants into a "new_state" dict per the configuration.
        new_state = {}
        for i, gen_out in enumerate(self.gen_outputs):
            new_state[gen_out] = results[i]

        # Prepare a de-chunked state dict which will be used for the injectors & losses.
        local_state = {}
        for k, v in state.items():
            local_state[k] = v[grad_accum_step]
        local_state.update(new_state)

        # Inject in any extra dependencies.
        for inj in self.injectors:
            injected = inj(local_state)
            local_state.update(injected)
            new_state.update(injected)

        if backward:
            # Finally, compute the losses.
            total_loss = 0
            for loss_name, loss in self.losses.items():
                l = loss(self.training_net, local_state)
                self.loss_accumulator.add_loss(loss_name, l)
                total_loss += l * self.weights[loss_name]
            self.loss_accumulator.add_loss("total", total_loss)

            # Get dem grads!
            with amp.scale_loss(total_loss, self.optimizers, amp_loss_id) as scaled_loss:
                scaled_loss.backward()

        return new_state


    # Performs the optimizer step after all gradient accumulation is completed. Default implementation simply steps()
    # all self.optimizers.
    def do_step(self):
        for opt in self.optimizers:
            opt.step()

    def get_metrics(self):
        return self.loss_accumulator.as_dict()
