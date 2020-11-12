import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.checkpoint import checkpoint_sequential

from models.archs.arch_util import make_layer, default_init_weights, ConvGnSilu, ConvGnLelu
from models.archs.srg2_classic import Interpolate
from utils.util import checkpoint


class ResidualDenseBlock(nn.Module):
    def __init__(self, mid_channels=64, growth_channels=32):
        super(ResidualDenseBlock, self).__init__()
        for i in range(5):
            out_channels = mid_channels if i == 4 else growth_channels
            self.add_module(
                f'conv{i+1}',
                nn.Conv2d(mid_channels + i * growth_channels, out_channels, 3,
                          1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        for i in range(5):
            default_init_weights(getattr(self, f'conv{i+1}'), 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


# Linear block wrapper with custom weights and lrelu activation suited for use with AdaIN.
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.lrelu = nn.LeakyReLU(.2)

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale)
        # Biased and scaled leaky relu.
        lrelu_bias = self.bias * self.lr_mul
        lrelu_dim = [1] * (out.ndim - lrelu_bias.ndim - 1)
        lrelu_scale = 2 ** .5
        out = self.lrelu(out + lrelu_bias.view(1, lrelu_bias.shape[0], *lrelu_dim)) * lrelu_scale
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class AdaIn(nn.Module):
    def __init__(self, channels, latent_nf):
        super(AdaIn, self).__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.latent_reducer = nn.Linear(latent_nf, channels * 2)
        self.channels = channels

    def forward(self, x, latent):
        xn = self.norm(x)
        latent = self.latent_reducer(latent)
        latent_bias = latent[:, :self.channels].view(x.shape[0], self.channels, 1,  1)
        latent_scale = latent[:, -self.channels:].view(x.shape[0], self.channels, 1, 1)
        return xn * latent_scale + latent_bias


class RRDBWithAdaIn(nn.Module):
    def __init__(self, mid_channels, growth_channels=32, latent_nf=256):
        super(RRDBWithAdaIn, self).__init__()
        self.adain1 = AdaIn(mid_channels, latent_nf)
        self.rdb1 = ResidualDenseBlock(mid_channels, growth_channels)
        self.adain2 = AdaIn(mid_channels, latent_nf)
        self.rdb2 = ResidualDenseBlock(mid_channels, growth_channels)
        self.adain3 = AdaIn(mid_channels, latent_nf)
        self.rdb3 = ResidualDenseBlock(mid_channels, growth_channels)

    def forward(self, x, latent):
        out = self.rdb1(self.adain1(x, latent))
        out = self.rdb2(self.adain2(out, latent))
        out = self.rdb3(self.adain3(out, latent))
        residual = out * .2
        return residual + x, residual


class ConvLatentEncoder(nn.Module):
    def __init__(self, latent_size):
        super(ConvLatentEncoder, self).__init__()
        layers = [EqualLinear(latent_size, latent_size, lr_mul=.01) for _ in range(8)]
        self.stack = nn.Sequential(*layers)

    def forward(self, latent):
        return self.stack(latent)


class AdaRRDBNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_blocks=23,
                 growth_channels=32,
                 blocks_per_checkpoint=4,
                 scale=4,
                 bottom_latent_only=False):
        super(AdaRRDBNet, self).__init__()
        self.latent_encoder = ConvLatentEncoder(256)
        self.num_blocks = num_blocks
        self.blocks_per_checkpoint = blocks_per_checkpoint
        self.scale = scale
        self.in_channels = in_channels
        self.nf = mid_channels
        self.bottom_latent_only = bottom_latent_only
        first_conv_stride = 1 if in_channels <= 4 else scale
        first_conv_ksize = 3 if first_conv_stride == 1 else 7
        first_conv_padding = 1 if first_conv_stride == 1 else 3
        self.conv_first = nn.Conv2d(in_channels, mid_channels, first_conv_ksize, first_conv_stride, first_conv_padding)
        self.body = make_layer(
            RRDBWithAdaIn,
            num_blocks,
            mid_channels=mid_channels,
            growth_channels=growth_channels,
            latent_nf=256)
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        for m in [
            self.conv_first, self.conv_body, self.conv_up1,
            self.conv_up2, self.conv_hr, self.conv_last
        ]:
            default_init_weights(m, 0.1)
        self.latent_mean = 0
        self.latent_std = 0
        self.latent_var = 0
        self.block_residual_means = []
        self.block_residual_stds = []

    def forward(self, x, latent=None, ref=None):
        latent_was_none = latent
        if latent is None:
            latent = torch.randn((x.shape[0], 256), device=x.device)
        latent = self.latent_encoder(latent)
        if latent_was_none is not None:
            self.latent_mean = torch.mean(latent).detach().cpu()
            self.latent_std = torch.std(latent).detach().cpu()
            self.latent_var = torch.var(latent).detach().cpu()
        if self.in_channels > 4:
            x_lg = F.interpolate(x, scale_factor=self.scale, mode="bicubic")
            if ref is None:
                ref = torch.zeros_like(x_lg)
            x_lg = torch.cat([x_lg, ref], dim=1)
        else:
            x_lg = x
        feat = self.conv_first(x_lg)
        body_feat = feat
        self.block_residual_means = []
        self.block_residual_stds = []
        for bl in self.body:
            body_feat, residual = checkpoint(bl, body_feat, latent)
            self.block_residual_means.append(torch.mean(residual).cpu())
            self.block_residual_stds.append(torch.std(residual).cpu())
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale == 4:
            feat = self.lrelu(
                self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        else:
            feat = self.lrelu(self.conv_up2(feat))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

    def get_debug_values(self, s, n):
        blk_stds, blk_means = {}, {}
        for i, (s, m) in enumerate(zip(self.block_residual_stds, self.block_residual_means)):
            blk_stds['block_%i' % (i+1,)] = s
            blk_means['block_%i' % (i+1,)] = m
        return {'encoded_latent_mean': self.latent_mean,
                'encoded_latent_std': self.latent_std,
                'encoded_latent_var': self.latent_var,
                'blocks_mean': blk_means,
                'blocks_std': blk_stds}


class LinearLatentEstimator(nn.Module):
    def __init__(self, in_nc, nf):
        super(LinearLatentEstimator, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [256, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [256, 4, 4]
        self.conv5_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn5_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv5_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(nf * 8, affine=True)

        self.bottom_channels = nf * 8 * 2 * 2
        self.l = nn.Linear(self.bottom_channels, 1024)
        self.l2 = nn.Linear(1024, 256)

        self.lrelu = nn.LeakyReLU(.2, inplace=True)
        self.norm = nn.LayerNorm(256)

    def compute_body(self, x):
        fea = self.lrelu(self.bn1_0(self.conv1_0(x)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))
        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
        fea = self.lrelu(self.bn5_0(self.conv5_0(fea)))
        fea = self.lrelu(self.bn5_1(self.conv5_1(fea)))

        return fea

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))
        o = checkpoint(self.compute_body, fea)
        o = o.view(o.shape[0], self.bottom_channels)
        o = self.lrelu(self.l(o))
        return self.norm(self.l2(o))

