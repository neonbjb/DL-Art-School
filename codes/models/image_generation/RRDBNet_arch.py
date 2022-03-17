import functools
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import Bottleneck

from models.arch_util import make_layer, default_init_weights, ConvGnSilu, ConvGnLelu
from trainer.networks import register_model
from utils.util import checkpoint, sequential_checkpoint, opt_get


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels=64, growth_channels=32, init_weight=.1):
        super(ResidualDenseBlock, self).__init__()
        for i in range(5):
            out_channels = mid_channels if i == 4 else growth_channels
            self.add_module(
                f'conv{i+1}',
                nn.Conv2d(mid_channels + i * growth_channels, out_channels, 3,
                          1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        for i in range(5):
            default_init_weights(getattr(self, f'conv{i+1}'), init_weight)


    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels, growth_channels=32, reduce_to=None):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(mid_channels, growth_channels)
        if reduce_to is not None:
            self.reducer = ConvGnLelu(mid_channels, reduce_to, kernel_size=3, activation=False, norm=False, bias=True)
            self.recover_ch = mid_channels - reduce_to
        else:
            self.reducer = None

    def forward(self, x, return_residual=False):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        if self.reducer is not None:
            out = self.reducer(out)
            b, f, h, w = out.shape
            out = torch.cat([out, torch.zeros((b, self.recover_ch, h, w), device=out.device)], dim=1)

        if return_residual:
            return 0.2 * out
        else:
            # Empirically, we use 0.2 to scale the residual for better performance
            return out * 0.2 + x


class RRDBWithBypass(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels, growth_channels=32, reduce_to=None, randomly_add_noise_to_bypass=False):
        super(RRDBWithBypass, self).__init__()
        self.rdb1 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(mid_channels, growth_channels)
        if reduce_to is not None:
            self.reducer = ConvGnLelu(mid_channels, reduce_to, kernel_size=3, activation=False, norm=False, bias=True)
            self.recover_ch = mid_channels - reduce_to
            bypass_channels = mid_channels + reduce_to
        else:
            self.reducer = None
            bypass_channels = mid_channels * 2
        self.bypass = nn.Sequential(ConvGnSilu(bypass_channels, mid_channels, kernel_size=3, bias=True, activation=True, norm=True),
                                    ConvGnSilu(mid_channels, mid_channels//2, kernel_size=3, bias=False, activation=True, norm=False),
                                    ConvGnSilu(mid_channels//2, 1, kernel_size=3, bias=False, activation=False, norm=False),
                                    nn.Sigmoid())
        self.randomly_add_bypass_noise = randomly_add_noise_to_bypass

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)

        if self.reducer is not None:
            out = self.reducer(out)
            b, f, h, w = out.shape
            out = torch.cat([out, torch.zeros((b, self.recover_ch, h, w), device=out.device)], dim=1)

        bypass = self.bypass(torch.cat([x, out], dim=1))
        # The purpose of random noise is to induce usage of bypass maps that would otherwise be "dead". Theoretically
        # if these maps provide value, the noise should trigger gradients to flow into the bypass conv network again.
        if self.randomly_add_bypass_noise and random.random() < .2:
            rnoise = torch.rand_like(bypass) * .02
            bypass = (bypass + rnoise).clamp(0, 1)
        self.bypass_map = bypass.detach().clone()

        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 * bypass + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Currently, it supports x4 upsampling scale factor.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Defaults: 23
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_blocks=23,
                 growth_channels=32,
                 body_block=RRDB,
                 blocks_per_checkpoint=1,
                 scale=4,
                 additive_mode="not",  # Options: "not", "additive", "additive_enforced"
                 headless=False,
                 feature_channels=64,  # Only applicable when headless=True. How many channels are used at the trunk level.
                 output_mode="hq_only",  # Options: "hq_only", "hq+features", "features_only"
                 initial_stride=1,
                 use_ref=False,  # When set, a reference image is expected as input and synthesized if not found. Useful for video SR.
                 ):
        super(RRDBNet, self).__init__()
        assert output_mode in ['hq_only', 'hq+features', 'features_only']
        assert additive_mode in ['not', 'additive', 'additive_enforced']
        self.num_blocks = num_blocks
        self.blocks_per_checkpoint = blocks_per_checkpoint
        self.scale = scale
        self.in_channels = in_channels
        self.output_mode = output_mode
        self.use_ref = use_ref
        first_conv_stride = initial_stride if not self.use_ref else scale
        first_conv_ksize = 3 if first_conv_stride == 1 else 7
        first_conv_padding = 1 if first_conv_stride == 1 else 3
        if headless:
            self.conv_first = None
            self.reduce_ch = feature_channels
            reduce_to = feature_channels
            self.conv_ref_first = ConvGnLelu(3, feature_channels, 7, stride=2, norm=False, activation=False, bias=True)
        else:
            self.conv_first = nn.Conv2d(in_channels, mid_channels, first_conv_ksize, first_conv_stride, first_conv_padding)
            self.reduce_ch = mid_channels
            reduce_to = None
        self.body = make_layer(
            body_block,
            num_blocks,
            mid_channels=mid_channels,
            growth_channels=growth_channels,
            reduce_to=reduce_to)
        self.conv_body = nn.Conv2d(self.reduce_ch, self.reduce_ch, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(self.reduce_ch, self.reduce_ch, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(self.reduce_ch, self.reduce_ch, 3, 1, 1)
        if scale >= 8:
            self.conv_up3 = nn.Conv2d(self.reduce_ch, self.reduce_ch, 3, 1, 1)
        else:
            self.conv_up3 = None
        self.conv_hr = nn.Conv2d(self.reduce_ch, self.reduce_ch, 3, 1, 1)
        self.conv_last = nn.Conv2d(self.reduce_ch, out_channels, 3, 1, 1)

        self.additive_mode = additive_mode
        if additive_mode == "additive_enforced":
            self.add_enforced_pool = nn.AvgPool2d(kernel_size=scale, stride=scale)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        for m in [
            self.conv_first, self.conv_body, self.conv_up1,
            self.conv_up2, self.conv_up3, self.conv_hr, self.conv_last
        ]:
            if m is not None:
                default_init_weights(m, 0.1)

    def forward(self, x, ref=None):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.conv_first is None:
            # Headless mode -> embedding inputs.
            if ref is not None:
                ref = self.conv_ref_first(ref)
                feat = torch.cat([x, ref], dim=1)
            else:
                feat = x
        else:
            # "Normal" mode -> image input.
            if self.use_ref:
                x_lg = F.interpolate(x, scale_factor=self.scale, mode="bicubic")
                if ref is None:
                    ref = torch.zeros_like(x_lg)
                x_lg = torch.cat([x_lg, ref], dim=1)
            else:
                x_lg = x
            feat = self.conv_first(x_lg)
        feat = sequential_checkpoint(self.body, self.num_blocks // self.blocks_per_checkpoint, feat)
        feat = feat[:, :self.reduce_ch]
        body_feat = self.conv_body(feat)
        feat = feat + body_feat
        if self.output_mode == "features_only":
            return feat

        # upsample
        out = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale >= 4:
            out = self.lrelu(
                self.conv_up2(F.interpolate(out, scale_factor=2, mode='nearest')))
            if self.scale >= 8:
                out = self.lrelu(
                    self.conv_up3(F.interpolate(out, scale_factor=2, mode='nearest')))
        else:
            out = self.lrelu(self.conv_up2(out))
        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        if "additive" in self.additive_mode:
            x_interp = F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        if self.additive_mode == 'additive':
            out = out + x_interp
        elif self.additive_mode == 'additive_enforced':
            out_pooled = self.add_enforced_pool(out)
            out = out - F.interpolate(out_pooled, scale_factor=self.scale, mode='nearest')
            out = out + x_interp

        if self.output_mode == "hq+features":
            return out, feat
        return out

    def visual_dbg(self, step, path):
        for i, bm in enumerate(self.body):
            if hasattr(bm, 'bypass_map'):
                torchvision.utils.save_image(bm.bypass_map.cpu().float(), os.path.join(path, "%i_bypass_%i.png" % (step, i+1)))

@register_model
def register_RRDBNetBypass(opt_net, opt):
    additive_mode = opt_net['additive_mode'] if 'additive_mode' in opt_net.keys() else 'not'
    output_mode = opt_net['output_mode'] if 'output_mode' in opt_net.keys() else 'hq_only'
    gc = opt_net['gc'] if 'gc' in opt_net.keys() else 32
    initial_stride = opt_net['initial_stride'] if 'initial_stride' in opt_net.keys() else 1
    bypass_noise = opt_get(opt_net, ['bypass_noise'], False)
    block = functools.partial(RRDBWithBypass, randomly_add_noise_to_bypass=bypass_noise)
    return RRDBNet(in_channels=opt_net['in_nc'], out_channels=opt_net['out_nc'],
                                mid_channels=opt_net['nf'], num_blocks=opt_net['nb'], additive_mode=additive_mode,
                                output_mode=output_mode, body_block=block, scale=opt_net['scale'], growth_channels=gc,
                                initial_stride=initial_stride)


@register_model
def register_RRDBNet(opt_net, opt):
    additive_mode = opt_net['additive_mode'] if 'additive_mode' in opt_net.keys() else 'not'
    output_mode = opt_net['output_mode'] if 'output_mode' in opt_net.keys() else 'hq_only'
    gc = opt_net['gc'] if 'gc' in opt_net.keys() else 32
    initial_stride = opt_net['initial_stride'] if 'initial_stride' in opt_net.keys() else 1
    return RRDBNet(in_channels=opt_net['in_nc'], out_channels=opt_net['out_nc'],
                                mid_channels=opt_net['nf'], num_blocks=opt_net['nb'], additive_mode=additive_mode,
                                output_mode=output_mode, body_block=RRDB, scale=opt_net['scale'], growth_channels=gc,
                                initial_stride=initial_stride)

