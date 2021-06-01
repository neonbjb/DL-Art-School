import logging
import os

import torch
from torch.nn.parallel import DataParallel
import torch.nn as nn

import trainer.lr_scheduler as lr_scheduler
import trainer.networks as networks
from trainer.base_model import BaseModel
from trainer.inject import create_injector
from trainer.steps import ConfigurableStep
from trainer.experiments.experiments import get_experiment_for_name
import torchvision.utils as utils

from utils.util import opt_get

logger = logging.getLogger('base')


class ExtensibleTrainer(BaseModel):
    def __init__(self, opt, cached_networks={}):
        super(ExtensibleTrainer, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # env is used as a global state to store things that subcomponents might need.
        self.env = {'device': self.device,
               'rank': self.rank,
               'opt': opt,
               'step': 0,
               'dist': opt['dist']
        }
        if opt['path']['models'] is not None:
               self.env['base_path'] = os.path.join(opt['path']['models'])

        self.mega_batch_factor = 1
        if self.is_train:
            self.mega_batch_factor = train_opt['mega_batch_factor']
            self.env['mega_batch_factor'] = self.mega_batch_factor
            self.batch_factor = self.mega_batch_factor
        self.checkpointing_cache = opt['checkpointing_enabled']

        self.netsG = {}
        self.netsD = {}
        # Note that this is on the chopping block. It should be integrated into an injection point.
        self.netF = networks.define_F().to(self.device)  # Used to compute feature loss.
        for name, net in opt['networks'].items():
            # Trainable is a required parameter, but the default is simply true. Set it here.
            if 'trainable' not in net.keys():
                net['trainable'] = True

            if name in cached_networks.keys():
                new_net = cached_networks[name]
            else:
                new_net = None
            if net['type'] == 'generator':
                if new_net is None:
                    new_net = networks.create_model(opt, net, self.netsG).to(self.device)
                self.netsG[name] = new_net
            elif net['type'] == 'discriminator':
                if new_net is None:
                    new_net = networks.create_model(opt, net, self.netsD).to(self.device)
                self.netsD[name] = new_net
            else:
                raise NotImplementedError("Can only handle generators and discriminators")

            if not net['trainable']:
                new_net.eval()
            if net['wandb_debug']:
                import wandb
                wandb.watch(new_net, log='all', log_freq=3)

        # Initialize the train/eval steps
        self.step_names = []
        self.steps = []
        for step_name, step in opt['steps'].items():
            step = ConfigurableStep(step, self.env)
            self.step_names.append(step_name)  # This could be an OrderedDict, but it's a PITA to integrate with AMP below.
            self.steps.append(step)

        # step.define_optimizers() relies on the networks being placed in the env, so put them there. Even though
        # they aren't wrapped yet.
        self.env['generators'] = self.netsG
        self.env['discriminators'] = self.netsD

        # Define the optimizers from the steps
        for s in self.steps:
            s.define_optimizers()
            self.optimizers.extend(s.get_optimizers())

        if self.is_train:
            # Find the optimizers that are using the default scheduler, then build them.
            def_opt = []
            for s in self.steps:
                def_opt.extend(s.get_optimizers_with_default_scheduler())
            self.schedulers = lr_scheduler.get_scheduler_for_name(train_opt['default_lr_scheme'], def_opt, train_opt)

            # Set the starting step count for the scheduler.
            for sched in self.schedulers:
                sched.last_epoch = opt['current_step']
        else:
            self.schedulers = []

        # Wrap networks in distributed shells.
        dnets = []
        all_networks = [g for g in self.netsG.values()] + [d for d in self.netsD.values()]
        for anet in all_networks:
            if opt['dist']:
                if opt['dist_backend'] == 'apex':
                    # Use Apex to enable delay_allreduce, which is compatible with gradient checkpointing.
                    from apex.parallel import DistributedDataParallel
                    dnet = DistributedDataParallel(anet, delay_allreduce=True)
                else:
                    from torch.nn.parallel.distributed import DistributedDataParallel
                    dnet = DistributedDataParallel(anet, device_ids=[torch.cuda.current_device()])
            else:
                dnet = DataParallel(anet, device_ids=opt['gpu_ids'])
            if self.is_train:
                dnet.train()
            else:
                dnet.eval()
            dnets.append(dnet)
        if not opt['dist']:
            self.netF = DataParallel(self.netF, device_ids=opt['gpu_ids'])

        # Backpush the wrapped networks into the network dicts..
        self.networks = {}
        found = 0
        for dnet in dnets:
            for net_dict in [self.netsD, self.netsG]:
                for k, v in net_dict.items():
                    if v == dnet.module:
                        net_dict[k] = dnet
                        self.networks[k] = dnet
                        found += 1
        assert found == len(self.netsG) + len(self.netsD)

        # Replace the env networks with the wrapped networks
        self.env['generators'] = self.netsG
        self.env['discriminators'] = self.netsD

        self.print_network()  # print network
        self.load()  # load G and D if needed

        # Load experiments
        self.experiments = []
        if 'experiments' in opt.keys():
            self.experiments = [get_experiment_for_name(e) for e in op['experiments']]

        # Setting this to false triggers SRGAN to call the models update_model() function on the first iteration.
        self.updated = True

    def feed_data(self, data, step, need_GT=True):
        self.env['step'] = step
        self.batch_factor = self.mega_batch_factor
        self.opt['checkpointing_enabled'] = self.checkpointing_cache
        # The batch factor can be adjusted on a period to allow known high-memory steps to fit in GPU memory.
        if 'mod_batch_factor' in self.opt['train'].keys() and \
                self.env['step'] % self.opt['train']['mod_batch_factor_every'] == 0:
            self.batch_factor = self.opt['train']['mod_batch_factor']
            if self.opt['train']['mod_batch_factor_also_disable_checkpointing']:
                self.opt['checkpointing_enabled'] = False

        self.eval_state = {}
        for o in self.optimizers:
            o.zero_grad()
        torch.cuda.empty_cache()

        self.dstate = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                self.dstate[k] = [t.to(self.device) for t in torch.chunk(v, chunks=self.batch_factor, dim=0)]

    def optimize_parameters(self, step):
        # Some models need to make parametric adjustments per-step. Do that here.
        for net in self.networks.values():
            if hasattr(net.module, "update_for_step"):
                net.module.update_for_step(step, os.path.join(self.opt['path']['models'], ".."))

        # Iterate through the steps, performing them one at a time.
        state = self.dstate
        for step_num, s in enumerate(self.steps):
            train_step = True
            # 'every' is used to denote steps that should only occur at a certain integer factor rate. e.g. '2' occurs every 2 steps.
            # Note that the injection points for the step might still be required, so address this by setting train_step=False
            if 'every' in s.step_opt.keys() and step % s.step_opt['every'] != 0:
                train_step = False
            # Steps can opt out of early (or late) training, make sure that happens here.
            if 'after' in s.step_opt.keys() and step < s.step_opt['after'] or 'before' in s.step_opt.keys() and step > s.step_opt['before']:
                continue
            # Steps can choose to not execute if a state key is missing.
            if 'requires' in s.step_opt.keys():
                requirements_met = True
                for requirement in s.step_opt['requires']:
                    if requirement not in state.keys():
                        requirements_met = False
                if not requirements_met:
                    continue

            if train_step:
                # Only set requires_grad=True for the network being trained.
                nets_to_train = s.get_networks_trained()
                enabled = 0
                for name, net in self.networks.items():
                    net_enabled = name in nets_to_train
                    if net_enabled:
                        enabled += 1
                    # Networks can opt out of training before a certain iteration by declaring 'after' in their definition.
                    if 'after' in self.opt['networks'][name].keys() and step < self.opt['networks'][name]['after']:
                        net_enabled = False
                    for p in net.parameters():
                        do_not_train_flag = hasattr(p, "DO_NOT_TRAIN") or (hasattr(p, "DO_NOT_TRAIN_UNTIL") and step < p.DO_NOT_TRAIN_UNTIL)
                        if p.dtype != torch.int64 and p.dtype != torch.bool and not do_not_train_flag:
                            p.requires_grad = net_enabled
                        else:
                            p.requires_grad = False
                assert enabled == len(nets_to_train)

                # Update experiments
                [e.before_step(self.opt, self.step_names[step_num], self.env, nets_to_train, state) for e in self.experiments]

                for o in s.get_optimizers():
                    o.zero_grad()

            # Now do a forward and backward pass for each gradient accumulation step.
            new_states = {}
            for m in range(self.batch_factor):
                ns = s.do_forward_backward(state, m, step_num, train=train_step)
                for k, v in ns.items():
                    if k not in new_states.keys():
                        new_states[k] = [v]
                    else:
                        new_states[k].append(v)

            # Push the detached new state tensors into the state map for use with the next step.
            for k, v in new_states.items():
                # State is immutable to reduce complexity. Overwriting existing state keys is not supported.
                class OverwrittenStateError(Exception):
                    def __init__(self, k, keys):
                        super().__init__(f'Attempted to overwrite state key: {k}.  The state should be considered '
                                         f'immutable and keys should not be overwritten. Current keys: {keys}')
                if k in state.keys():
                    raise OverwrittenStateError(k, list(state.keys()))
                state[k] = v

            if train_step:
                # And finally perform optimization.
                [e.before_optimize(state) for e in self.experiments]
                s.do_step(step)
                # Some networks have custom steps, for example EMA
                for net in self.networks:
                    if hasattr(net, "custom_optimizer_step"):
                        net.custom_optimizer_step(step)
                [e.after_optimize(state) for e in self.experiments]

        # Record visual outputs for usage in debugging and testing.
        if 'visuals' in self.opt['logger'].keys() and self.rank <= 0 and step % self.opt['logger']['visual_debug_rate'] == 0:
            denorm = 'image_normalization_range' in self.opt.keys()
            denorm_range = opt_get(self.opt, ['image_normalization_range'], None)
            if denorm_range:
                denorm_range = tuple(denorm_range)
            sample_save_path = os.path.join(self.opt['path']['models'], "..", "visual_dbg")
            for v in self.opt['logger']['visuals']:
                if v not in state.keys():
                    continue   # This can happen for several reasons (ex: 'after' defs), just ignore it.
                for i, dbgv in enumerate(state[v]):
                    if 'recurrent_visual_indices' in self.opt['logger'].keys() and len(dbgv.shape)==5:
                        for rvi in self.opt['logger']['recurrent_visual_indices']:
                            rdbgv = dbgv[:, rvi]
                            if rdbgv.shape[1] > 3:
                                rdbgv = rdbgv[:, :3, :, :]
                            os.makedirs(os.path.join(sample_save_path, v), exist_ok=True)
                            utils.save_image(rdbgv.float(), os.path.join(sample_save_path, v, "%05i_%02i_%02i.png" % (step, rvi, i)), normalize=denorm, range=denorm_range)
                    else:
                        if dbgv.shape[1] > 3:
                            dbgv = dbgv[:,:3,:,:]
                        os.makedirs(os.path.join(sample_save_path, v), exist_ok=True)
                        utils.save_image(dbgv.float(), os.path.join(sample_save_path, v, "%05i_%02i.png" % (step, i)), normalize=denorm, range=denorm_range)
            # Some models have their own specific visual debug routines.
            for net_name, net in self.networks.items():
                if hasattr(net.module, "visual_dbg"):
                    model_vdbg_dir = os.path.join(sample_save_path, net_name)
                    os.makedirs(model_vdbg_dir, exist_ok=True)
                    net.module.visual_dbg(step, model_vdbg_dir)

    def compute_fea_loss(self, real, fake):
        with torch.no_grad():
            logits_real = self.netF(real.to(self.device))
            logits_fake = self.netF(fake.to(self.device))
        return nn.L1Loss().to(self.device)(logits_fake, logits_real)

    def test(self):
        for net in self.netsG.values():
            net.eval()

        with torch.no_grad():
            # This can happen one of two ways: Either a 'validation injector' is provided, in which case we run that.
            # Or, we run the entire chain of steps in "train" mode and use eval.output_state.
            if 'injectors' in self.opt['eval'].keys():
                state = {}
                for inj in self.opt['eval']['injectors'].values():
                    # Need to move from mega_batch mode to batch mode (remove chunks)
                    for k, v in self.dstate.items():
                        state[k] = v[0]
                    inj = create_injector(inj, self.env)
                    state.update(inj(state))
            else:
                # Iterate through the steps, performing them one at a time.
                state = self.dstate
                for step_num, s in enumerate(self.steps):
                    ns = s.do_forward_backward(state, 0, step_num, train=False)
                    for k, v in ns.items():
                        state[k] = [v]

            self.eval_state = {}
            for k, v in state.items():
                if isinstance(v, list):
                    self.eval_state[k] = [s.detach().cpu() if isinstance(s, torch.Tensor) else s for s in v]
                else:
                    self.eval_state[k] = [v.detach().cpu() if isinstance(v, torch.Tensor) else v]

        for net in self.netsG.values():
            net.train()

    # Fetches a summary of the log.
    def get_current_log(self, step):
        log = {}
        for s in self.steps:
            log.update(s.get_metrics())

        for e in self.experiments:
            log.update(e.get_log_data())

        # Some generators can do their own metric logging.
        for net_name, net in self.networks.items():
            if hasattr(net.module, "get_debug_values"):
                log.update(net.module.get_debug_values(step, net_name))

        # Log learning rate (from first param group) too.
        for o in self.optimizers:
            for pgi, pg in enumerate(o.param_groups):
                log['learning_rate_%s_%i' % (o._config['network'], pgi)] = pg['lr']
        return log

    def get_current_visuals(self, need_GT=True):
        # Conforms to an archaic format from MMSR.
        res = {'lq': self.eval_state['lq'][0].float().cpu(),
               'rlt': self.eval_state[self.opt['eval']['output_state']][0].float().cpu()}
        if 'hq' in self.eval_state.keys():
            res['hq'] = self.eval_state['hq'][0].float().cpu(),
        return res

    def print_network(self):
        for name, net in self.networks.items():
            s, n = self.get_network_description(net)
            net_struc_str = '{}'.format(net.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network {} structure: {}, with parameters: {:,d}'.format(name, net_struc_str, n))
                logger.info(s)

    def load(self):
        for netdict in [self.netsG, self.netsD]:
            for name, net in netdict.items():
                if not self.opt['networks'][name]['trainable']:
                    continue
                load_path = self.opt['path']['pretrain_model_%s' % (name,)]
                if load_path is not None:
                    if self.rank <= 0:
                        logger.info('Loading model for [%s]' % (load_path,))
                    self.load_network(load_path, net, self.opt['path']['strict_load'], opt_get(self.opt, ['path', f'pretrain_base_path_{name}']))
                if hasattr(net.module, 'network_loaded'):
                    net.module.network_loaded()

    def save(self, iter_step):
        for name, net in self.networks.items():
            # Don't save non-trainable networks.
            if self.opt['networks'][name]['trainable']:
                self.save_network(net, name, iter_step)

    def force_restore_swapout(self):
        # Legacy method. Do nothing.
        pass
