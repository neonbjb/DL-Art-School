import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.archs.srflow.FlowUpsamplerNet import FlowUpsamplerNet
import models.archs.srflow.thops as thops
import models.archs.srflow.flow as flow
from models.archs.srflow.RRDBNet_arch import RRDBNet


class SRFlowNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, quant, flow_block_maps, noise_quant,
                 hidden_channels=64, gc=32, scale=4, K=16, L=3, train_rrdb_at_step=0,
                 hr_img_shape=(128,128,3), coupling='CondAffineSeparatedAndCond'):
        super(SRFlowNet, self).__init__()

        self.scale = scale
        self.noise_quant = noise_quant
        self.quant = quant
        self.flow_block_maps = flow_block_maps
        self.RRDB = RRDBNet(in_nc, out_nc, nf, nb, gc, scale, flow_block_maps)
        self.train_rrdb_step = train_rrdb_at_step
        self.RRDB_training = True

        self.flowUpsamplerNet = FlowUpsamplerNet(image_shape=hr_img_shape,
                                                 hidden_channels=hidden_channels,
                                                 scale=scale, rrdb_blocks=flow_block_maps,
                                                 K=K, L=L, flow_coupling=coupling)
        self.i = 0

    def forward(self, gt=None, lr=None, reverse=False, z=None, eps_std=None, epses=None, reverse_with_grad=False,
                lr_enc=None,
                add_gt_noise=False, step=None, y_label=None):
        if not reverse:
            return self.normal_flow(gt, lr, epses=epses, lr_enc=lr_enc, add_gt_noise=add_gt_noise, step=step,
                                    y_onehot=y_label)
        else:
            assert lr.shape[1] == 3
            if reverse_with_grad:
                return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                         add_gt_noise=add_gt_noise)
            else:
                with torch.no_grad():
                    return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                             add_gt_noise=add_gt_noise)

    def normal_flow(self, gt, lr, y_onehot=None, epses=None, lr_enc=None, add_gt_noise=True, step=None):
        if lr_enc is None:
            lr_enc = self.rrdbPreprocessing(lr)

        logdet = torch.zeros_like(gt[:, 0, 0, 0])
        pixels = thops.pixels(gt)

        z = gt

        if add_gt_noise:
            # Setup
            if self.noise_quant:
                z = z + ((torch.rand(z.shape, device=z.device) - 0.5) / self.quant)
            logdet = logdet + float(-np.log(self.quant) * pixels)

        # Encode
        epses, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, gt=z, logdet=logdet, reverse=False, epses=epses,
                                              y_onehot=y_onehot)

        objective = logdet.clone()

        if isinstance(epses, (list, tuple)):
            z = epses[-1]
        else:
            z = epses

        objective = objective + flow.GaussianDiag.logp(None, None, z)

        nll = (-objective) / float(np.log(2.) * pixels)

        if isinstance(epses, list):
            return epses, nll, logdet
        return z, nll, logdet

    def rrdbPreprocessing(self, lr):
        rrdbResults = self.RRDB(lr, get_steps=True)
        block_idxs = self.flow_block_maps
        if len(block_idxs) > 0:
            concat = torch.cat([rrdbResults["block_{}".format(idx)] for idx in block_idxs], dim=1)

            keys = ['last_lr_fea', 'fea_up1', 'fea_up2', 'fea_up4']
            if 'fea_up0' in rrdbResults.keys():
                keys.append('fea_up0')
            if 'fea_up-1' in rrdbResults.keys():
                keys.append('fea_up-1')
            if self.scale >= 8:
                keys.append('fea_up8')
            if self.scale == 16:
                keys.append('fea_up16')
            for k in keys:
                h = rrdbResults[k].shape[2]
                w = rrdbResults[k].shape[3]
                rrdbResults[k] = torch.cat([rrdbResults[k], F.interpolate(concat, (h, w))], dim=1)
        return rrdbResults

    def get_score(self, disc_loss_sigma, z):
        score_real = 0.5 * (1 - 1 / (disc_loss_sigma ** 2)) * thops.sum(z ** 2, dim=[1, 2, 3]) - \
                     z.shape[1] * z.shape[2] * z.shape[3] * math.log(disc_loss_sigma)
        return -score_real

    def reverse_flow(self, lr, z, y_onehot, eps_std, epses=None, lr_enc=None, add_gt_noise=True):
        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        pixels = thops.pixels(lr) * self.scale ** 2

        if add_gt_noise:
            logdet = logdet - float(-np.log(self.quant) * pixels)

        if lr_enc is None:
            lr_enc = self.rrdbPreprocessing(lr)

        x, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, z=z, eps_std=eps_std, reverse=True, epses=epses,
                                          logdet=logdet)

        return x, logdet

    def set_rrdb_training(self, trainable):
        if self.RRDB_training != trainable:
            for p in self.RRDB.parameters():
                if not trainable:
                    p.DO_NOT_TRAIN = True
                elif hasattr(p, "DO_NOT_TRAIN"):
                    del p.DO_NOT_TRAIN
            self.RRDB_training = trainable

    def update_for_step(self, step, experiments_path='.'):
        self.set_rrdb_training(step > self.train_rrdb_step)