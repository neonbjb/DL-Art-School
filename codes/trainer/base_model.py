import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel

import utils.util


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.amp_level = 'O0' if opt['amp_opt_level'] is None else opt['amp_opt_level']
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        self.disc_optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler"""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.last_epoch = cur_iter
            scheduler.step()
        # set up warm-up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        return str(network), sum(map(lambda x: x.numel(), network.parameters()))

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
        # Also save to the 'alt_path' which is useful for caching to Google Drive in colab, for example.
        if 'alt_path' in self.opt['path'].keys():
            torch.save(state_dict, os.path.join(self.opt['path']['alt_path'], save_filename))
        if self.opt['colab_mode']:
            utils.util.copy_files_to_server(self.opt['ssh_server'], self.opt['ssh_username'], self.opt['ssh_password'],
                                            save_path, os.path.join(self.opt['remote_path'], 'models', save_filename))
        return save_path

    def load_network(self, load_path, network, strict=True, pretrain_base_path=None):
        # Sometimes networks are passed in as DDP modules, we want the raw parameters.
        if hasattr(network, 'module'):
            network = network.module
        load_net = torch.load(load_path)

        # Support loading torch.save()s for whole models as well as just state_dicts.
        if 'state_dict' in load_net:
            load_net = load_net['state_dict']
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'

        if pretrain_base_path is not None:
            t = load_net
            load_net = {}
            for k, v in t.items():
                if k.startswith(pretrain_base_path):
                    load_net[k[len(pretrain_base_path):]] = v

        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k.replace('module.', '')] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        """Save training state during training, which will be used for resuming"""
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        if 'amp_opt_level' in self.opt.keys():
            state['amp'] = amp.state_dict()
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)
        # Also save to the 'alt_path' which is useful for caching to Google Drive in colab, for example.
        if 'alt_path' in self.opt['path'].keys():
            torch.save(state, os.path.join(self.opt['path']['alt_path'], 'latest.state'))
        if self.opt['colab_mode']:
            utils.util.copy_files_to_server(self.opt['ssh_server'], self.opt['ssh_username'], self.opt['ssh_password'],
                                            save_path, os.path.join(self.opt['remote_path'], 'training_state', save_filename))

    def resume_training(self, resume_state, load_amp=True):
        """Resume the optimizers and schedulers for training"""
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
        if load_amp and 'amp' in resume_state.keys():
            from apex import amp
            amp.load_state_dict(resume_state['amp'])
