import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.image_generation.srflow.RRDBNet_arch import RRDBNet
from models.image_generation.srflow.FlowUpsamplerNet import FlowUpsamplerNet
import models.image_generation.srflow.thops as thops
import models.image_generation.srflow.flow as flow
from trainer.networks import register_model
from utils.util import opt_get


class SRFlowNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, K=None, opt=None, step=None):
        super(SRFlowNet, self).__init__()

        self.opt = opt
        self.quant = 255 if opt_get(opt, ['datasets', 'train', 'quant']) is \
                            None else opt_get(opt, ['datasets', 'train', 'quant'])
        initial_stride = opt_get(opt, ['networks', 'generator', 'initial_stride'], 1)
        self.RRDB = RRDBNet(in_nc, out_nc, nf=nf, nb=nb, gc=gc, scale=scale, opt=opt, initial_conv_stride=initial_stride)
        if 'pretrain_rrdb' in opt['networks']['generator'].keys():
            rrdb_state_dict = torch.load(opt['networks']['generator']['pretrain_rrdb'])
            self.RRDB.load_state_dict(rrdb_state_dict, strict=True)

        hidden_channels = opt_get(opt, ['networks', 'generator','flow', 'hidden_channels'])
        hidden_channels = hidden_channels or 64
        self.RRDB_training = opt_get(self.opt, ['networks', 'generator','train_RRDB'], default=False)
        self.flow_scale = opt_get(self.opt, ['networks', 'generator', 'flow_scale'], default=opt['scale'])

        self.patch_sz = opt_get(self.opt, ['networks', 'generator', 'flow', 'patch_size'], 160)
        self.flowUpsamplerNet = \
            FlowUpsamplerNet((self.patch_sz, self.patch_sz, 3), hidden_channels, K,
                             flow_coupling=opt['networks']['generator']['flow']['coupling'], opt=opt)
        self.i = 0
        self.dbg_logp = 0
        self.dbg_logdet = 0

    def get_random_z(self, heat, seed=None, batch_size=1, lr_shape=None, device='cuda'):
        if seed: torch.manual_seed(seed)
        if opt_get(self.opt, ['networks', 'generator', 'flow', 'split', 'enable']):
            C = self.flowUpsamplerNet.C
            H = int(self.flow_scale * lr_shape[0] // (self.flowUpsamplerNet.scaleH * self.flow_scale / self.RRDB.scale))
            W = int(self.flow_scale * lr_shape[1] // (self.flowUpsamplerNet.scaleW * self.flow_scale / self.RRDB.scale))

            size = (batch_size, C, H, W)
            if heat == 0:
                z = torch.zeros(size)
            else:
                z = torch.normal(mean=0, std=heat, size=size)
        else:
            L = opt_get(self.opt, ['networks', 'generator', 'flow', 'L']) or 3
            fac = 2 ** (L - 3)
            z_size = int(self.lr_size // (2 ** (L - 3)))
            z = torch.normal(mean=0, std=heat, size=(batch_size, 3 * 8 * 8 * fac * fac, z_size, z_size))
        return z.to(device)

    def forward(self, gt=None, lr=None, z=None, eps_std=None, reverse=False, epses=None, reverse_with_grad=False,
                lr_enc=None,
                add_gt_noise=True, step=None, y_label=None):
        if not reverse:
            return self.normal_flow(gt, lr, epses=epses, lr_enc=lr_enc, add_gt_noise=add_gt_noise, step=step,
                                    y_onehot=y_label)
        else:
            assert lr.shape[1] == 3
            if z is None:
                # Synthesize it. Accommodate mismatches in LR scale and flow_scale, which are normally handled by the RRDB subnet.
                lr_shape = [d * self.opt['scale'] / self.flow_scale for d in lr.shape[2:]]
                z = self.get_random_z(eps_std, batch_size=lr.shape[0], lr_shape=lr_shape, device=lr.device)
            if reverse_with_grad:
                return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                         add_gt_noise=add_gt_noise)
            else:
                with torch.no_grad():
                    return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                             add_gt_noise=add_gt_noise)

    def normal_flow(self, gt, lr, y_onehot=None, epses=None, lr_enc=None, add_gt_noise=True, step=None):
        if lr_enc is None:
            if self.RRDB_training:
                lr_enc = self.rrdbPreprocessing(lr)
            else:
                with torch.no_grad():
                    lr_enc = self.rrdbPreprocessing(lr)

        logdet = torch.zeros_like(gt[:, 0, 0, 0])
        pixels = thops.pixels(gt)

        z = gt

        if add_gt_noise:
            # Setup
            noiseQuant = opt_get(self.opt, ['networks', 'generator','flow', 'augmentation', 'noiseQuant'], True)
            if noiseQuant:
                z = z + ((torch.rand(z.shape, device=z.device) - 0.5) / self.quant)
            logdet = logdet + float(-np.log(self.quant) * pixels)

        # Encode
        epses, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, gt=z, logdet=logdet, reverse=False, epses=[],
                                              y_onehot=y_onehot)

        objective = logdet.clone()

        if isinstance(epses, (list, tuple)):
            z = epses[-1]
        else:
            z = epses

        logp = flow.GaussianDiag.logp(None, None, z)
        objective = objective + logp

        nll = (-objective) / float(np.log(2.) * pixels)
        self.dbg_logp = -logp.mean().item() / float(np.log(2.) * pixels)
        self.dbg_logdet = -logdet.mean().item() / float(np.log(2.) * pixels)

        if isinstance(epses, list):
            return epses, nll, logdet
        return z, nll, logdet

    def get_debug_values(self, s, n):
        return {"logp": self.dbg_logp, "logdet": self.dbg_logdet}

    def rrdbPreprocessing(self, lr):
        rrdbResults = self.RRDB(lr, get_steps=True)
        block_idxs = opt_get(self.opt, ['networks', 'generator', 'flow', 'stackRRDB', 'blocks']) or []
        if len(block_idxs) > 0:
            concat = torch.cat([rrdbResults["block_{}".format(idx)] for idx in block_idxs], dim=1)

            if opt_get(self.opt, ['networks', 'generator','flow', 'stackRRDB', 'concat']) or False:
                keys = ['last_lr_fea', 'fea_up1', 'fea_up2', 'fea_up4']
                if 'fea_up0' in rrdbResults.keys():
                    keys.append('fea_up0')
                if 'fea_up-1' in rrdbResults.keys():
                    keys.append('fea_up-1')
                if self.flow_scale >= 8:
                    keys.append('fea_up8')
                if self.flow_scale == 16:
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
        pixels = thops.pixels(lr) * self.opt['scale'] ** 2

        if add_gt_noise:
            logdet = logdet - float(-np.log(self.quant) * pixels)

        if lr_enc is None:
            if self.RRDB_training:
                lr_enc = self.rrdbPreprocessing(lr)
            else:
                with torch.no_grad():
                    lr_enc = self.rrdbPreprocessing(lr)

        x, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, z=z, eps_std=eps_std, reverse=True, epses=epses,
                                          logdet=logdet)

        return x, logdet, lr_enc['out']


@register_model
def register_srflow(opt_net, opt):
    return SRFlowNet(in_nc=3, out_nc=3, nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['scale'],
                             K=opt_net['K'], opt=opt)
