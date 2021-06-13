from torch.cuda.amp import GradScaler

from utils.loss_accumulator import LossAccumulator
from torch.nn import Module
import logging
from trainer.losses import create_loss
import torch
from collections import OrderedDict
from trainer.inject import create_injector
from utils.util import recursively_detach, opt_get

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
        self.scaler = GradScaler(enabled=self.opt['fp16'])
        self.grads_generated = False
        self.min_total_loss = opt_step['min_total_loss'] if 'min_total_loss' in opt_step.keys() else -999999999

        self.injectors = []
        if 'injectors' in self.step_opt.keys():
            injector_names = []
            for inj_name, injector in self.step_opt['injectors'].items():
                assert inj_name not in injector_names  # Repeated names are always an error case.
                injector_names.append(inj_name)
                self.injectors.append(create_injector(injector, env))

        losses = []
        self.weights = {}
        if 'losses' in self.step_opt.keys():
            for loss_name, loss in self.step_opt['losses'].items():
                assert loss_name not in self.weights.keys()  # Repeated names are always an error case.
                losses.append((loss_name, create_loss(loss, env)))
                self.weights[loss_name] = loss['weight']
        self.losses = OrderedDict(losses)

    def get_network_for_name(self, name):
        return self.env['generators'][name] if name in self.env['generators'].keys() \
                else self.env['discriminators'][name]

    # Subclasses should override this to define individual optimizers. They should all go into self.optimizers.
    #  This default implementation defines a single optimizer for all Generator parameters.
    #  Must be called after networks are initialized and wrapped.
    def define_optimizers(self):
        opt_configs = [opt_get(self.step_opt, ['optimizer_params'], None)]
        self.optimizers = []
        if opt_configs[0] is None:
            return
        training = self.step_opt['training']
        training_net = self.get_network_for_name(training)
        nets = [training_net]
        training = [training]
        for net_name, net, opt_config in zip(training, nets, opt_configs):
            # Configs can organize parameters by-group and specify different learning rates for each group. This only
            # works in the model specifically annotates which parameters belong in which group using PARAM_GROUP.
            optim_params = {'default': {'params': [], 'lr': opt_config['lr']}}
            if opt_config is not None and 'param_groups' in opt_config.keys():
                for k, pg in opt_config['param_groups'].items():
                    optim_params[k] = {'params': [], 'lr': pg['lr']}

            for k, v in net.named_parameters():  # can optimize for a part of the model
                # Make some inference about these parameters, which can be used by some optimizers to treat certain
                # parameters differently. For example, it is considered good practice to not do weight decay on
                # BN & bias parameters. TODO: process the module tree instead of the parameter tree to accomplish the
                # same thing, but in a more effective way.
                if k.endswith(".bias"):
                    v.is_bias = True
                if k.endswith(".weight"):
                    v.is_weight = True
                if ".bn" in k or '.batchnorm' in k or '.bnorm' in k:
                    v.is_bn = True
                # Some models can specify some parameters to be in different groups.
                param_group = "default"
                if hasattr(v, 'PARAM_GROUP'):
                    if v.PARAM_GROUP in optim_params.keys():
                        param_group = v.PARAM_GROUP
                    else:
                        logger.warning(f'Model specifies a custom param group {v.PARAM_GROUP} which is not configured. '
                                       f'The same LR will be used for all parameters.')

                if v.requires_grad:
                    optim_params[param_group]['params'].append(v)
                else:
                    if self.env['rank'] <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            if 'optimizer' not in self.step_opt.keys() or self.step_opt['optimizer'] == 'adam':
                opt = torch.optim.Adam(list(optim_params.values()),
                                       weight_decay=opt_config['weight_decay'],
                                       betas=(opt_config['beta1'], opt_config['beta2']))
            elif self.step_opt['optimizer'] == 'adamw':
                opt = torch.optim.AdamW(list(optim_params.values()),
                                       weight_decay=opt_config['weight_decay'],
                                       betas=(opt_config['beta1'], opt_config['beta2']))
            elif self.step_opt['optimizer'] == 'lars':
                from trainer.optimizers.larc import LARC
                from trainer.optimizers.sgd import SGDNoBiasMomentum
                optSGD = SGDNoBiasMomentum(list(optim_params.values()), lr=opt_config['lr'], momentum=opt_config['momentum'],
                                           weight_decay=opt_config['weight_decay'])
                opt = LARC(optSGD, trust_coefficient=opt_config['lars_coefficient'])
            elif self.step_opt['optimizer'] == 'sgd':
                from torch.optim import SGD
                opt = SGD(list(optim_params.values()), lr=opt_config['lr'], momentum=opt_config['momentum'], weight_decay=opt_config['weight_decay'])
            opt._config = opt_config  # This is a bit seedy, but we will need these configs later.
            opt._config['network'] = net_name
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

    # Performs all forward and backward passes for this step given an input state. All input states are lists of
    # chunked tensors. Use grad_accum_step to dereference these steps. Should return a dict of tensors that later
    # steps might use. These tensors are automatically detached and accumulated into chunks.
    def do_forward_backward(self, state, grad_accum_step, amp_loss_id, train=True):
        local_state = {}  # <-- Will store the entire local state to be passed to injectors & losses.
        new_state = {}  # <-- Will store state values created by this step for returning to ExtensibleTrainer.
        for k, v in state.items():
            local_state[k] = v[grad_accum_step]
        local_state['train_nets'] = str(self.get_networks_trained())

        # Some losses compute backward() internally. Accommodate this by stashing the amp_loss_id in env.
        self.env['amp_loss_id'] = amp_loss_id
        self.env['current_step_optimizers'] = self.optimizers
        self.env['training'] = train

        # Inject in any extra dependencies.
        for inj in self.injectors:
            # Don't do injections tagged with eval unless we are not in train mode.
            if train and 'eval' in inj.opt.keys() and inj.opt['eval']:
                continue
            # Likewise, don't do injections tagged with train unless we are not in eval.
            if not train and 'train' in inj.opt.keys() and inj.opt['train']:
                continue
            # Don't do injections tagged with 'after' or 'before' when we are out of spec.
            if 'after' in inj.opt.keys() and self.env['step'] < inj.opt['after'] or \
               'before' in inj.opt.keys() and self.env['step'] > inj.opt['before'] or \
               'every' in inj.opt.keys() and self.env['step'] % inj.opt['every'] != 0:
                continue
            if 'no_accum' in inj.opt.keys() and grad_accum_step > 0:
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
                if 'after' in loss.opt.keys() and loss.opt['after'] > self.env['step'] or \
                   'before' in loss.opt.keys() and self.env['step'] > loss.opt['before'] or \
                   'every' in loss.opt.keys() and self.env['step'] % loss.opt['every'] != 0:
                    continue
                if loss.is_stateful():
                    l, lstate = loss(self.get_network_for_name(self.step_opt['training']), local_state)
                    local_state.update(lstate)
                    new_state.update(lstate)
                else:
                    l = loss(self.get_network_for_name(self.step_opt['training']), local_state)
                total_loss += l * self.weights[loss_name]
                # Record metrics.
                if isinstance(l, torch.Tensor):
                    self.loss_accumulator.add_loss(loss_name, l)
                for n, v in loss.extra_metrics():
                    self.loss_accumulator.add_loss("%s_%s" % (loss_name, n), v)
                    loss.clear_metrics()

            # In some cases, the loss could not be set (e.g. all losses have 'after')
            if isinstance(total_loss, torch.Tensor):
                self.loss_accumulator.add_loss("%s_total" % (self.get_training_network_name(),), total_loss)
                reset_required = total_loss < self.min_total_loss

                # Scale the loss down by the accumulation factor.
                total_loss = total_loss / self.env['mega_batch_factor']

                # Get dem grads!
                self.scaler.scale(total_loss).backward()

                if reset_required:
                    # You might be scratching your head at this. Why would you zero grad as opposed to not doing a
                    # backwards? Because DDP uses the backward() pass as a synchronization point and there is not a good
                    # way to simply bypass backward. If you want a more efficient way to specify a min_loss, use or
                    # implement it at the loss level.
                    self.get_network_for_name(self.step_opt['training']).zero_grad()
                    self.loss_accumulator.increment_metric("%s_skipped_steps" % (self.get_training_network_name(),))

                self.grads_generated = True

        # Detach all state variables. Within the step, gradients can flow. Once these variables leave the step
        # we must release the gradients.
        new_state = recursively_detach(new_state)
        return new_state

    # Performs the optimizer step after all gradient accumulation is completed. Default implementation simply steps()
    # all self.optimizers.
    def do_step(self, step):
        if not self.grads_generated:
            return
        self.grads_generated = False
        for opt in self.optimizers:
            # Optimizers can be opted out in the early stages of training.
            after = opt._config['after'] if 'after' in opt._config.keys() else 0
            after_network = self.opt['networks'][opt._config['network']]['after'] if 'after' in self.opt['networks'][opt._config['network']].keys() else 0
            after = max(after, after_network)
            if self.env['step'] < after:
                continue
            before = opt._config['before'] if 'before' in opt._config.keys() else -1
            if before != -1 and self.env['step'] > before:
                continue
            self.scaler.step(opt)
            self.scaler.update()

    def get_metrics(self):
        return self.loss_accumulator.as_dict()
