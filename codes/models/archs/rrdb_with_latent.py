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
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

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


    def forward(self, x, identity=None):
        if identity is None:
            identity = x
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + identity


class RRDBWithBypassAndLatent(nn.Module):
    def __init__(self, mid_channels, growth_channels=32, latent_dim=256):
        super(RRDBWithBypassAndLatent, self).__init__()
        self.latent_process = nn.Sequential(nn.Linear(latent_dim, latent_dim//2, bias=False),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                            nn.Linear(latent_dim//2, mid_channels, bias=False),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                            nn.Linear(mid_channels, mid_channels, bias=True))
        self.latent_join = nn.Sequential(ConvGnLelu(mid_channels*2, mid_channels*2, activation=True, norm=False, bias=False),
                                         ConvGnLelu(mid_channels*2, mid_channels, activation=False, norm=False, bias=False))
        self.rdb1 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(mid_channels, growth_channels)
        self.bypass = nn.Sequential(ConvGnSilu(mid_channels*2, mid_channels, kernel_size=3, bias=True, activation=True, norm=True),
                                    ConvGnSilu(mid_channels, mid_channels//2, kernel_size=3, bias=False, activation=True, norm=False),
                                    ConvGnSilu(mid_channels//2, 1, kernel_size=3, bias=False, activation=False, norm=False),
                                    nn.Sigmoid())

    def forward(self, x, original_latent):
        b, f, h, w = x.shape
        latent = self.latent_process(original_latent)
        b, l = latent.shape
        latent = latent.view(b, l, 1, 1)
        latent = latent.repeat(1, 1, h, w)
        out = self.latent_join(torch.cat([x, latent], dim=1))
        out = self.rdb1(out, x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        bypass = self.bypass(torch.cat([x, out], dim=1))
        self.bypass_map = bypass.detach().clone()
        residual = out * .2 * bypass
        return residual + x, residual


class RRDBNetWithLatent(nn.Module):
    # 8-layer MLP in the vein of StyleGAN.
    def create_linear_latent_encoder(self, latent_size):
        return nn.Sequential(nn.Linear(latent_size, latent_size),
                            nn.BatchNorm1d(latent_size),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Linear(latent_size, latent_size),
                            nn.BatchNorm1d(latent_size),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Linear(latent_size, latent_size),
                            nn.BatchNorm1d(latent_size),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Linear(latent_size, latent_size),
                            nn.BatchNorm1d(latent_size),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Linear(latent_size, latent_size),
                            nn.BatchNorm1d(latent_size),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Linear(latent_size, latent_size),
                            nn.BatchNorm1d(latent_size),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Linear(latent_size, latent_size),
                            nn.BatchNorm1d(latent_size),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Linear(latent_size, latent_size),
                            nn.BatchNorm1d(latent_size),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    # Creates a 2D latent by iterating through the provided latent_filters and doubling the
    # image size each step.
    def create_conv_latent_encoder(self, latent_filters):
        layers = []
        for i in range(len(latent_filters)-1):
            layers.extend(ConvGnLelu(latent_filters[i], latent_filters[i]))
            layers.extend(Interpolate(2))
            layers.extend(ConvGnLelu(latent_filters[i], latent_filters[i+1]))
        return nn.Sequential(*layers)

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_blocks=23,
                 growth_channels=32,
                 blocks_per_checkpoint=4,
                 scale=4,
                 latent_size=256):
        super(RRDBNetWithLatent, self).__init__()
        self.num_blocks = num_blocks
        self.blocks_per_checkpoint = blocks_per_checkpoint
        self.scale = scale
        self.in_channels = in_channels
        self.latent_size = latent_size
        first_conv_stride = 1 if in_channels <= 4 else scale
        first_conv_ksize = 3 if first_conv_stride == 1 else 7
        first_conv_padding = 1 if first_conv_stride == 1 else 3
        self.conv_first = nn.Conv2d(in_channels, mid_channels, first_conv_ksize, first_conv_stride, first_conv_padding)
        self.body = make_layer(
            RRDBWithBypassAndLatent,
            num_blocks,
            mid_channels=mid_channels,
            growth_channels=growth_channels,
            latent_dim=latent_size)
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 8-layer MLP in the vein of StyleGAN.
        self.latent_encoder = nn.Sequential(nn.Linear(latent_size, latent_size),
                                            nn.BatchNorm1d(latent_size),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                            nn.Linear(latent_size, latent_size),
                                            nn.BatchNorm1d(latent_size),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                            nn.Linear(latent_size, latent_size),
                                            nn.BatchNorm1d(latent_size),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                            nn.Linear(latent_size, latent_size),
                                            nn.BatchNorm1d(latent_size),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                            nn.Linear(latent_size, latent_size),
                                            nn.BatchNorm1d(latent_size),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                            nn.Linear(latent_size, latent_size),
                                            nn.BatchNorm1d(latent_size),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                            nn.Linear(latent_size, latent_size),
                                            nn.BatchNorm1d(latent_size),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                            nn.Linear(latent_size, latent_size),
                                            nn.BatchNorm1d(latent_size),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        for m in [
            self.conv_first, self.conv_body, self.conv_up1,
            self.conv_up2, self.conv_hr, self.conv_last
        ]:
            default_init_weights(m, 0.1)

    def forward(self, x, latent=None, ref=None):
        latent_was_none = latent
        if latent is None:
            latent = torch.randn((x.shape[0], self.latent_size), dtype=torch.float, device=x.device)
        latent = checkpoint(self.latent_encoder, latent)
        if latent_was_none is None:
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

    def visual_dbg(self, step, path):
        for i, bm in enumerate(self.body):
            torchvision.utils.save_image(bm.bypass_map.cpu().float(), os.path.join(path, "%i_bypass_%i.png" % (step, i+1)))

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


# Based heavily on the same VGG arch used for the discriminator.
class LatentEstimator(nn.Module):
    # input_img_factor = multiplier to support images over 128x128. Only certain factors are supported.
    def __init__(self, in_nc, nf, latent_size=256):
        super(LatentEstimator, self).__init__()
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
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
        final_nf = nf * 8

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.linear1 = nn.Linear(int(final_nf * 4 * 4), latent_size*2)
        self.linear2 = nn.Linear(latent_size*2, latent_size)
        self.tanh = nn.Tanh()

    def compute_body(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        #fea = torch.cat([fea, skip_med], dim=1)
        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        #fea = torch.cat([fea, skip_lo], dim=1)
        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
        return fea

    def forward(self, x):
        fea = checkpoint(self.compute_body, x)
        fea = fea.contiguous().view(fea.size(0), -1)
        fea = self.linear1(fea)
        out = self.tanh(self.linear2(fea))
        self.latent_mean = torch.mean(out)
        self.latent_std = torch.std(out)
        self.latent_var = torch.var(out)
        return out

    def get_debug_values(self, s, n):
        return {'latent_estimator_mean': self.latent_mean,
                'latent_estimator_std': self.latent_std,
                'latent_estimator_var': self.latent_var}

