import torch.nn as nn
import torch.nn.functional as F

from models.archs.RRDBNet_arch import RRDB
from models.archs.arch_util import make_layer, default_init_weights, ConvGnSilu, ConvGnLelu, PixelUnshuffle
from utils.util import checkpoint, sequential_checkpoint


class MultiLevelRRDB(nn.Module):
    def __init__(self, nf, gc, levels):
        super().__init__()
        self.levels = levels
        self.level_rrdbs = nn.ModuleList([RRDB(nf, growth_channels=gc) for i in range(levels)])

    # Trunks should be fed in in order HR->LR
    def forward(self, trunk):
        for i in reversed(range(self.levels)):
            lvl_scale = (2**i)
            lvl_res = self.level_rrdbs[i](F.interpolate(trunk, scale_factor=1/lvl_scale, mode="area"), return_residual=True)
            trunk = trunk + F.interpolate(lvl_res, scale_factor=lvl_scale, mode="nearest")
        return trunk


class MultiResRRDBNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 l1_blocks=3,
                 l2_blocks=4,
                 l3_blocks=6,
                 growth_channels=32,
                 scale=4,
                 ):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 7, stride=1, padding=3)

        self.l3_blocks = nn.ModuleList([MultiLevelRRDB(mid_channels, growth_channels, 3) for _ in range(l1_blocks)])
        self.l2_blocks = nn.ModuleList([MultiLevelRRDB(mid_channels, growth_channels, 2) for _ in range(l2_blocks)])
        self.l1_blocks = nn.ModuleList([MultiLevelRRDB(mid_channels, growth_channels, 1) for _ in range(l3_blocks)])
        self.block_levels = [self.l3_blocks, self.l2_blocks, self.l1_blocks]

        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        for m in [
            self.conv_first, self.conv_first, self.conv_body, self.conv_up1,
            self.conv_up2, self.conv_hr, self.conv_last
        ]:
            if m is not None:
                default_init_weights(m, 0.1)

    def forward(self, x):
        trunk = self.conv_first(x)
        for block_set in self.block_levels:
            for block in block_set:
                trunk = checkpoint(block, trunk)

        body_feat = self.conv_body(trunk)
        feat = trunk + body_feat

        # upsample
        out = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale == 4:
            out = self.lrelu(
                self.conv_up2(F.interpolate(out, scale_factor=2, mode='nearest')))
        else:
            out = self.lrelu(self.conv_up2(out))
        out = self.conv_last(self.lrelu(self.conv_hr(out)))

        return out

    def visual_dbg(self, step, path):
        pass


class SteppedResRRDBNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 l1_blocks=3,
                 l2_blocks=3,
                 growth_channels=32,
                 scale=4,
                 ):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 7, stride=2, padding=3)
        self.conv_second = nn.Conv2d(mid_channels, mid_channels*2, 3, stride=2, padding=1)

        self.l1_blocks = nn.Sequential(*[RRDB(mid_channels*2, growth_channels*2) for _ in range(l1_blocks)])
        self.l1_upsample_conv = nn.Conv2d(mid_channels*2, mid_channels, 3, stride=1, padding=1)
        self.l2_blocks = nn.Sequential(*[RRDB(mid_channels, growth_channels, 2) for _ in range(l2_blocks)])

        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        for m in [
            self.conv_first, self.conv_second, self.l1_upsample_conv, self.conv_body, self.conv_up1,
            self.conv_up2, self.conv_hr, self.conv_last
        ]:
            if m is not None:
                default_init_weights(m, 0.1)

    def forward(self, x):
        trunk = self.conv_first(x)
        trunk = self.conv_second(trunk)
        trunk = sequential_checkpoint(self.l1_blocks, len(self.l2_blocks), trunk)
        trunk = F.interpolate(trunk, scale_factor=2, mode="nearest")
        trunk = self.l1_upsample_conv(trunk)
        trunk = sequential_checkpoint(self.l2_blocks, len(self.l2_blocks), trunk)
        body_feat = self.conv_body(trunk)
        feat = trunk + body_feat

        # upsample
        out = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale == 4:
            out = self.lrelu(
                self.conv_up2(F.interpolate(out, scale_factor=2, mode='nearest')))
        else:
            out = self.lrelu(self.conv_up2(out))
        out = self.conv_last(self.lrelu(self.conv_hr(out)))

        return out

    def visual_dbg(self, step, path):
        pass


class PixelShufflingSteppedResRRDBNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 l1_blocks=3,
                 l2_blocks=3,
                 growth_channels=32,
                 scale=2,
                 ):
        super().__init__()
        self.scale = scale * 2  # This RRDB operates at half-scale resolution.
        self.in_channels = in_channels

        self.pix_unshuffle = PixelUnshuffle(4)
        self.conv_first = nn.Conv2d(4*4*in_channels, mid_channels*2, 3, stride=1, padding=1)

        self.l1_blocks = nn.Sequential(*[RRDB(mid_channels*2, growth_channels*2) for _ in range(l1_blocks)])
        self.l1_upsample_conv = nn.Conv2d(mid_channels*2, mid_channels, 3, stride=1, padding=1)
        self.l2_blocks = nn.Sequential(*[RRDB(mid_channels, growth_channels, 2) for _ in range(l2_blocks)])

        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        for m in [
            self.conv_first, self.l1_upsample_conv, self.conv_body, self.conv_up1,
            self.conv_up2, self.conv_hr, self.conv_last
        ]:
            if m is not None:
                default_init_weights(m, 0.1)

    def forward(self, x):
        trunk = self.conv_first(self.pix_unshuffle(x))
        trunk = sequential_checkpoint(self.l1_blocks, len(self.l1_blocks), trunk)
        trunk = F.interpolate(trunk, scale_factor=2, mode="nearest")
        trunk = self.l1_upsample_conv(trunk)
        trunk = sequential_checkpoint(self.l2_blocks, len(self.l2_blocks), trunk)
        body_feat = self.conv_body(trunk)
        feat = trunk + body_feat

        # upsample
        out = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale == 4:
            out = self.lrelu(
                self.conv_up2(F.interpolate(out, scale_factor=2, mode='nearest')))
        else:
            out = self.lrelu(self.conv_up2(out))
        out = self.conv_last(self.lrelu(self.conv_hr(out)))

        return out

    def visual_dbg(self, step, path):
        pass
