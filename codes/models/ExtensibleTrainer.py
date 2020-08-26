import logging
import os

import torch
from apex import amp
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn as nn

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.base_model import BaseModel
from models.steps.steps import ConfigurableStep
import torchvision.utils as utils

logger = logging.getLogger('base')


class ExtensibleTrainer(BaseModel):
    def __init__(self, opt):
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
               'step': 0}

        self.netsG = {}
        self.netsD = {}
        self.netF = networks.define_F().to(self.device)  # Used to compute feature loss.
        for name, net in opt['networks'].items():
            if net['type'] == 'generator':
                new_net = networks.define_G(net, None, opt['scale']).to(self.device)
                self.netsG[name] = new_net
            elif net['type'] == 'discriminator':
                new_net = networks.define_D_net(net, opt['datasets']['train']['target_size']).to(self.device)
                self.netsD[name] = new_net
            else:
                raise NotImplementedError("Can only handle generators and discriminators")

        # Initialize the train/eval steps
        self.steps = []
        for step_name, step in opt['steps'].items():
            step = ConfigurableStep(step, self.env)
            self.steps.append(step)

        if self.is_train:
            self.mega_batch_factor = train_opt['mega_batch_factor']
            if self.mega_batch_factor is None:
                self.mega_batch_factor = 1
            self.env['mega_batch_factor'] = self.mega_batch_factor

            # The steps rely on the networks being placed in the env, so put them there. Even though they arent wrapped
            # yet.
            self.env['generators'] = self.netsG
            self.env['discriminators'] = self.netsD

            # Define the optimizers from the steps
            for s in self.steps:
                s.define_optimizers()
                self.optimizers.extend(s.get_optimizers())

            # Find the optimizers that are using the default scheduler, then build them.
            def_opt = []
            for s in self.steps:
                def_opt.extend(s.get_optimizers_with_default_scheduler())
            self.schedulers = lr_scheduler.get_scheduler_for_name(train_opt['default_lr_scheme'], def_opt, train_opt)

            # Initialize amp.
            total_nets = [g for g in self.netsG.values()] + [d for d in self.netsD.values()]
            amp_nets, amp_opts = amp.initialize(total_nets + [self.netF] + self.steps,
                                                self.optimizers, opt_level=opt['amp_opt_level'], num_losses=len(opt['steps']))

            # Unwrap steps & netF
            self.netF = amp_nets[len(total_nets)]
            assert(len(self.steps) == len(amp_nets[len(total_nets)+1:]))
            self.steps = amp_nets[len(total_nets)+1:]
            amp_nets = amp_nets[:len(total_nets)]

            # DataParallel
            dnets = []
            for anet in amp_nets:
                if opt['dist']:
                    dnet = DistributedDataParallel(anet,
                                                   device_ids=[torch.cuda.current_device()],
                                                   find_unused_parameters=True)
                else:
                    dnet = DataParallel(anet)
                if self.is_train:
                    dnet.train()
                else:
                    dnet.eval()
                dnets.append(dnet)
            if not opt['dist']:
                self.netF = DataParallel(self.netF)

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

        # Setting this to false triggers SRGAN to call the models update_model() function on the first iteration.
        self.updated = True

    def feed_data(self, data):
        self.lq = torch.chunk(data['LQ'].to(self.device), chunks=self.mega_batch_factor, dim=0)
        self.hq = [t.to(self.device) for t in torch.chunk(data['GT'], chunks=self.mega_batch_factor, dim=0)]
        input_ref = data['ref'] if 'ref' in data else data['GT']
        self.ref = [t.to(self.device) for t in torch.chunk(input_ref, chunks=self.mega_batch_factor, dim=0)]

        self.dstate = {'lq': self.lq, 'hq': self.hq, 'ref': self.ref}
        for k, v in data.items():
            if k not in ['LQ', 'ref', 'GT'] and isinstance(v, torch.Tensor):
                self.dstate[k] = [t.to(self.device) for t in torch.chunk(v, chunks=self.mega_batch_factor, dim=0)]

    def optimize_parameters(self, step):
        self.env['step'] = step

        # Some models need to make parametric adjustments per-step. Do that here.
        for net in self.networks.values():
            if hasattr(net.module, "update_for_step"):
                net.module.update_for_step(step, os.path.join(self.opt['path']['models'], ".."))

        # Iterate through the steps, performing them one at a time.
        state = self.dstate
        for step_num, s in enumerate(self.steps):
            # Only set requires_grad=True for the network being trained.
            nets_to_train = s.get_networks_trained()
            enabled = 0
            for name, net in self.networks.items():
                net_enabled = name in nets_to_train
                if net_enabled:
                    enabled += 1
                for p in net.parameters():
                    if p.dtype != torch.int64 and p.dtype != torch.bool:
                        p.requires_grad = net_enabled
                    else:
                        p.requires_grad = False
            assert enabled == len(nets_to_train)

            for o in s.get_optimizers():
                o.zero_grad()

            # Now do a forward and backward pass for each gradient accumulation step.
            new_states = {}
            for m in range(self.mega_batch_factor):
                ns = s.do_forward_backward(state, m, step_num)
                for k, v in ns.items():
                    if k not in new_states.keys():
                        new_states[k] = [v]
                    else:
                        new_states[k].append(v)

            # Push the detached new state tensors into the state map for use with the next step.
            for k, v in new_states.items():
                # State is immutable to reduce complexity. Overwriting existing state keys is not supported.
                assert k not in state.keys()
                state[k] = v

            # And finally perform optimization.
            s.do_step()

        # Record visual outputs for usage in debugging and testing.
        if 'visuals' in self.opt['logger'].keys():
            sample_save_path = os.path.join(self.opt['path']['models'], "..", "visual_dbg")
            for v in self.opt['logger']['visuals']:
                if step % self.opt['logger']['visual_debug_rate'] == 0:
                    for i, dbgv in enumerate(state[v]):
                        os.makedirs(os.path.join(sample_save_path, v), exist_ok=True)
                        utils.save_image(dbgv, os.path.join(sample_save_path, v, "%05i_%02i.png" % (step, i)))

    def compute_fea_loss(self, real, fake):
        with torch.no_grad():
            logits_real = self.netF(real)
            logits_fake = self.netF(fake)
        return nn.L1Loss().to(self.device)(logits_fake, logits_real)

    def test(self):
        for net in self.netsG.values():
            net.eval()

        with torch.no_grad():
            # Iterate through the steps, performing them one at a time.
            state = self.dstate
            for step_num, s in enumerate(self.steps):
                ns = s.do_forward_backward(state, 0, step_num, backward=False)
                for k, v in ns.items():
                    state[k] = [v]

            self.eval_state = state

        for net in self.netsG.values():
            net.train()

    # Fetches a summary of the log.
    def get_current_log(self, step):
        log = {}
        for s in self.steps:
            log.update(s.get_metrics())

        # Some generators can do their own metric logging.
        for net in self.networks.values():
            if hasattr(net.module, "get_debug_values"):
                log.update(net.module.get_debug_values(step))
        return log

    def get_current_visuals(self, need_GT=True):
        # Conforms to an archaic format from MMSR.
        return {'LQ': self.eval_state['lq'][0].float().cpu(),
                'GT': self.eval_state['hq'][0].float().cpu(),
                'rlt': self.eval_state[self.opt['eval']['output_state']][0].float().cpu()}

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
                load_path = self.opt['path']['pretrain_model_%s' % (name,)]
                if load_path is not None:
                    logger.info('Loading model for [%s]' % (load_path))
                    self.load_network(load_path, net)

    def save(self, iter_step):
        for name, net in self.networks.items():
            self.save_network(net, name, iter_step)

    def force_restore_swapout(self):
        # Legacy method. Do nothing.
        pass