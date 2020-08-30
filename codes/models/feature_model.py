import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel

logger = logging.getLogger('base')


class FeatureModel(BaseModel):
    def __init__(self, opt):
        super(FeatureModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        self.fea_train = networks.define_F(for_training=True).to(self.device)
        self.net_ref = networks.define_F().to(self.device)

        self.load()

        if self.is_train:
            self.fea_train.train()

            # loss
            self.cri_fea = nn.MSELoss().to(self.device)

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.fea_train.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['gen_lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        # grey out the LR image but keep 3 channels.
        lr = torch.mean(self.real_H, dim=1, keepdim=True)
        lr = lr.repeat(1, 3, 1, 1)

        self.fake_H = self.fea_train(lr, interpolate_factor=1)
        ref_H = self.net_ref(self.real_H)
        l_fea = self.cri_fea(self.fake_H, ref_H)
        l_fea.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_fea'] = l_fea.item()

    def test(self):
        pass

    def get_current_log(self, step):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        return None

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for F [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.fea_train, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.fea_train, 'G', iter_label)
