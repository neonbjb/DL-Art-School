from utils.loss_accumulator import LossAccumulator
from torch.nn import Module
import logging
from models.steps.losses import create_generator_loss
import torch
from apex import amp
from collections import OrderedDict
from .injectors import create_injector
from models.novograd import NovoGrad

logger = logging.getLogger('base')


# Defines the expected API for a single training step
class ConfigurableStep(Module):

    def __init__(self, opt_step, env):
        super(ConfigurableStep, self).__init__()

        self.step_opt = opt_step
        self.env = env
        self.opt = env['opt']
        self.gen_outputs = opt_step['generator_outputs']
        self.loss_accumulator = LossAccumulator()
        self.optimizers = None

        self.injectors = []
        if 'injectors' in self.step_opt.keys():
            for inj_name, injector in self.step_opt['injectors'].items():
                self.injectors.append(create_injector(injector, env))

        losses = []
        self.weights = {}
        if 'losses' in self.step_opt.keys():
            for loss_name, loss in self.step_opt['losses'].items():
                losses.append((loss_name, create_generator_loss(loss, env)))
                self.weights[loss_name] = loss['weight']
        self.losses = OrderedDict(losses)

    # Subclasses should override this to define individual optimizers. They should all go into self.optimizers.
    #  This default implementation defines a single optimizer for all Generator parameters.
    #  Must be called after networks are initialized and wrapped.
    def define_optimizers(self):
        self.training_net = self.env['generators'][self.step_opt['training']] \
            if self.step_opt['training'] in self.env['generators'].keys() \
            else self.env['discriminators'][self.step_opt['training']]
        optim_params = []
        for k, v in self.training_net.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.env['rank'] <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        if 'optimizer' not in self.step_opt.keys() or self.step_opt['optimizer'] == 'adam':
            opt = torch.optim.Adam(optim_params, lr=self.step_opt['lr'],
                                   weight_decay=self.step_opt['weight_decay'],
                                   betas=(self.step_opt['beta1'], self.step_opt['beta2']))
        elif self.step_opt['optimizer'] == 'novograd':
            opt = NovoGrad(optim_params, lr=self.step_opt['lr'], weight_decay=self.step_opt['weight_decay'],
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
    def do_forward_backward(self, state, grad_accum_step, amp_loss_id, train=True):
        new_state = {}

        # Prepare a de-chunked state dict which will be used for the injectors & losses.
        local_state = {}
        for k, v in state.items():
            local_state[k] = v[grad_accum_step]
        local_state.update(new_state)

        # Inject in any extra dependencies.
        for inj in self.injectors:
            # Don't do injections tagged with eval unless we are not in train mode.
            if train and 'eval' in inj.opt.keys() and inj.opt['eval']:
                continue
            # Likewise, don't do injections tagged with train unless we are not in eval.
            if not train and 'train' in inj.opt.keys() and inj.opt['train']:
                continue
            injected = inj(local_state)
            local_state.update(injected)
            new_state.update(injected)

        if train and len(self.losses) > 0:
            # Finally, compute the losses.
            total_loss = 0
            for loss_name, loss in self.losses.items():
                # Some losses only activate after a set number of steps. For example, proto-discriminator losses can
                # be very disruptive to a generator.
                if 'after' in loss.opt.keys() and loss.opt['after'] > self.env['step']:
                    continue

                l = loss(self.training_net, local_state)
                total_loss += l * self.weights[loss_name]
                # Record metrics.
                self.loss_accumulator.add_loss(loss_name, l)
                for n, v in loss.extra_metrics():
                    self.loss_accumulator.add_loss("%s_%s" % (loss_name, n), v)
            self.loss_accumulator.add_loss("%s_total" % (self.step_opt['training'],), total_loss)
            # Scale the loss down by the accumulation factor.
            total_loss = total_loss / self.env['mega_batch_factor']

            # Get dem grads!
            if self.env['amp']:
                with amp.scale_loss(total_loss, self.optimizers, amp_loss_id) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()

        # Detach all state variables. Within the step, gradients can flow. Once these variables leave the step
        # we must release the gradients.
        for k, v in new_state.items():
            if isinstance(v, torch.Tensor):
                new_state[k] = v.detach()
        return new_state

    # Performs the optimizer step after all gradient accumulation is completed. Default implementation simply steps()
    # all self.optimizers.
    def do_step(self):
        for opt in self.optimizers:
            opt.step()

    def get_metrics(self):
        return self.loss_accumulator.as_dict()
