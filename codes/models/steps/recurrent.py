from utils.loss_accumulator import LossAccumulator
from torch.nn import Module
import logging
from models.steps.losses import create_loss
import torch
from apex import amp
from collections import OrderedDict
from .injectors import create_injector
from models.novograd import NovoGrad
from utils.util import recursively_detach

logger = logging.getLogger('base')


def define_recurrent_controller(opt, env):
    pass


class RecurrentController:
    def __init__(self, opt, env):
        self.opt = opt
        self.env = env

    # This is the meat of the RecurrentController code. It is expected to return a recurrent_state which is fed into
    # the injectors and losses, or None if the recurrent loop is to be exited.
    # Note that on the first call, the recurrent_state parameter is set to None.
    def get_next_step(self, state, recurrent_state):
        return None


# This class implements the logic necessary to gather the gradients resulting from recurrent network passes.
class RecurrentStep(Module):

    def __init__(self, opt_step, env):
        super(RecurrentStep, self).__init__()

        self.step_opt = opt_step
        self.env = env
        self.opt = env['opt']
        self.gen_outputs = opt_step['generator_outputs']
        self.loss_accumulator = LossAccumulator()
        self.optimizers = None

        # Recurrent steps must have a bespoke "controller". This is a snippet of code responsible for determining
        # how many recurrent steps should be executed, and also compiles a "recurrent_state" which is passed to the
        # injectors and losses within the recurrent loop. Note that the recurrent state does not persist past the
        # recurrent loop.
        self.controller = define_recurrent_controller(self.step_opt)

        # Unlike a "normal" step, recurrent steps have 2 injection sites: "initial" and "recurrent". Initial injectors
        # are run once when the step is first executed. Recurrent injectors are run for every recurrent cycle and their
        # outputs are appended to a list.
        self.initial_injectors = []
        if 'initial_injectors' in self.step_opt.keys():
            for inj_name, injector in self.step_opt['initial_injectors'].items():
                self.initial_injectors.append(create_injector(injector, env))
        self.recurrent_injectors = []
        if 'recurrent_injectors' in self.step_opt.keys():
            for inj_name, injector in self.step_opt['recurrent_injectors'].items():
                self.recurrent_injectors.append(create_injector(injector, env))

        # Recurrent detach points are a list of state variables that get detached on every iteration. Since recurrent
        # injections are pushed into lists, detach points specify the exact tensor to detach by being a list of lists,
        # e.g.: [['var1', -2], ['var2', -1], ['var3', 0]]
        # The first element of the sublist is the state variable you want to detach. The second element is a list index
        # into that state variable.
        self.recurrent_detach_points = []
        if 'recurrent_detach_points' in self.step_opt.keys():
            for name, index in self.step_opt['recurrent_detach_points']:
                self.recurrent_detach_points.append(name,  index)

        # Recurrent steps also have two types of losses: 'recurrent' and 'final'.
        # Similar to injection points, 'recurrent' losses are invoked every iteration.
        # 'final' losses are invoked after all iterations have completed.
        losses = []
        self.recurrent_weights = {}
        if 'recurrent_losses' in self.step_opt.keys():
            for loss_name, loss in self.step_opt['recurrent_losses'].items():
                losses.append((loss_name, create_loss(loss, env)))
                self.recurrent_weights[loss_name] = loss['weight']
        self.recurrent_losses = OrderedDict(losses)
        self.final_weights = {}
        if 'final_losses' in self.step_opt.keys():
            for loss_name, loss in self.step_opt['final_losses'].items():
                losses.append((loss_name, create_loss(loss, env)))
                self.final_weights[loss_name] = loss['weight']
        self.final_losses = OrderedDict(losses)

    def get_network_for_name(self, name):
        return self.env['generators'][name] if name in self.env['generators'].keys() \
                else self.env['discriminators'][name]

    # Subclasses should override this to define individual optimizers. They should all go into self.optimizers.
    #  This default implementation defines a single optimizer for all Generator parameters.
    #  Must be called after networks are initialized and wrapped.
    def define_optimizers(self):
        training = self.step_opt['training']
        if isinstance(training, list):
            self.training_net = [self.get_network_for_name(t) for t in training]
            opt_configs = [self.step_opt['optimizer_params'][t] for t in training]
            nets = self.training_net
        else:
            self.training_net = self.get_network_for_name(training)
            # When only training one network, optimizer params can just embedded in the step params.
            if 'optimizer_params' not in self.step_opt.keys():
                opt_configs = [self.step_opt]
            else:
                opt_configs = [self.step_opt['optimizer_params']]
            nets = [self.training_net]
        self.optimizers = []
        for net, opt_config in zip(nets, opt_configs):
            optim_params = []
            for k, v in net.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.env['rank'] <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            if 'optimizer' not in self.step_opt.keys() or self.step_opt['optimizer'] == 'adam':
                opt = torch.optim.Adam(optim_params, lr=opt_config['lr'],
                                       weight_decay=opt_config['weight_decay'],
                                       betas=(opt_config['beta1'], opt_config['beta2']))
            elif self.step_opt['optimizer'] == 'novograd':
                opt = NovoGrad(optim_params, lr=opt_config['lr'], weight_decay=opt_config['weight_decay'],
                                       betas=(opt_config['beta1'], opt_config['beta2']))
            opt._config = opt_config  # This is a bit seedy, but we will need these configs later.
            self.optimizers.append(opt)

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
        if isinstance(self.step_opt['training'], list):
            return self.step_opt['training']
        else:
            return [self.step_opt['training']]

    def get_training_network_name(self):
        if isinstance(self.step_opt['training'], list):
            return self.step_opt['training'][0]
        else:
            return self.step_opt['training']

    def do_injection(self, injectors, local_state, train=True):
        injected_state = {}
        for inj in injectors:
            # Don't do injections tagged with eval unless we are not in train mode.
            if train and 'eval' in inj.opt.keys() and inj.opt['eval']:
                continue
            # Likewise, don't do injections tagged with train unless we are not in eval.
            if not train and 'train' in inj.opt.keys() and inj.opt['train']:
                continue
            # Don't do injections tagged with 'after' or 'before' when we are out of spec.
            if 'after' in inj.opt.keys() and self.env['step'] < inj.opt['after'] or \
                'before' in inj.opt.keys() and self.env['step'] > inj.opt['before']:
                continue
            injected_state.update(inj(local_state))
        return injected_state

    def compute_gradients(self, losses, weights, local_state, amp_loss_id):
        total_loss = 0
        for loss_name, loss in losses.items():
            # Some losses only activate after a set number of steps. For example, proto-discriminator losses can
            # be very disruptive to a generator.
            if 'after' in loss.opt.keys() and loss.opt['after'] > self.env['step']:
                continue

            l = loss(self.training_net, local_state)
            total_loss += l * weights[loss_name]
            # Record metrics.
            self.loss_accumulator.add_loss(loss_name, l)
            for n, v in loss.extra_metrics():
                self.loss_accumulator.add_loss("%s_%s" % (loss_name, n), v)
        self.loss_accumulator.add_loss("%s_total" % (self.get_training_network_name(),), total_loss)

        # Scale the loss down by the accumulation factor.
        total_loss = total_loss / self.env['mega_batch_factor']

        # Get dem grads!
        if self.env['amp']:
            with amp.scale_loss(total_loss, self.optimizers, amp_loss_id) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

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
        local_state['train_nets'] = str(self.get_networks_trained())

        # Some losses compute backward() internally. Accomodate this by stashing the amp_loss_id in env.
        self.env['amp_loss_id'] = amp_loss_id
        self.env['current_step_optimizers'] = self.optimizers
        self.env['training'] = train

        # Inject in initial tensors.
        injected = self.do_injection(self.initial_injectors, local_state, train)
        local_state.update(injected)
        new_state.update(injected)

        recurrent_state = self.controller.get_next_step(state, None)
        while recurrent_state:
            # Detach items no longer needed from previous recursive loop.
            for name, ind in self.recurrent_detach_points:
                len_required = ind if ind > 0 else abs(ind)+1
                if len(local_state[name]) >= len_required:
                    local_state[name][ind] = local_state[name][ind].detach()

            # Recurrent injectors and losses rely on state variables from recurrent_state. Combine that with local_state.
            combined_state = local_state
            combined_state.update(recurrent_state)

            # Inject recurrent injections.
            injected = self.do_injection(self.recurrent_injectors, combined_state, train)
            for k, v in injected.items():
                if k not in local_state.keys():
                    local_state[k] = []
                    combined_state[k] = []
                    new_state[k] = []
                local_state[k].append(v)
                combined_state[k].append(v)
                new_state[k].append(v.detach())

            # Compute the recurrent losses.
            if train:
                self.compute_gradients(self.recurrent_losses, self.recurrent_weights, combined_state, amp_loss_id)

            # Zero out combined_state, it'll be repopulated in the next loop.
            combined_state = {}

        # Compute the final losses
        if train:
            self.compute_gradients(self.final_losses, self.final_weights, local_state, amp_loss_id)

        # Detach all state variables. Within the step, gradients can flow. Once these variables leave the step
        # we must release the gradients.
        new_state = recursively_detach(new_state)
        return new_state

    # Performs the optimizer step after all gradient accumulation is completed. Default implementation simply steps()
    # all self.optimizers.
    def do_step(self):
        for opt in self.optimizers:
            # Optimizers can be opted out in the early stages of training.
            after = opt._config['after'] if 'after' in opt._config.keys() else 0
            if self.env['step'] < after:
                continue
            before = opt._config['before'] if 'before' in opt._config.keys() else -1
            if before != -1 and self.env['step'] > before:
                continue
            opt.step()

    def get_metrics(self):
        return self.loss_accumulator.as_dict()
