import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch


class HighToLowResNet(nn.Module):
    ''' ResNet that applies a noise channel to the input, then downsamples it four times using strides. Finally, the
     input is upsampled to the desired downscale. Currently downscale=1,2,4 is supported.
     '''
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, downscale=4):
        super(HighToLowResNet, self).__init__()

        assert downscale in [1, 2, 4], "Requested downscale not supported; %i" % (downscale, )
        self.downscale = downscale

        # We will always apply a noise channel to the inputs, account for that here.
        in_nc += 1

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        # All sub-modules must be explicit members. Make it so. Then add them to a list.
        self.trunk1 = arch_util.make_layer(functools.partial(arch_util.ResidualBlock_noBN, nf=nf), 4)
        self.trunk2 = arch_util.make_layer(functools.partial(arch_util.ResidualBlock_noBN, nf=nf*4), 8)
        self.trunk3 = arch_util.make_layer(functools.partial(arch_util.ResidualBlock_noBN, nf=nf*8), 16)
        self.trunk4 = arch_util.make_layer(functools.partial(arch_util.ResidualBlock_noBN, nf=nf*16), 32)
        self.trunks = [self.trunk1, self.trunk2, self.trunk3, self.trunk4]
        self.trunkshapes = [4, 8, 16, 32]

        self.r1 = nn.Conv2d(nf, nf*4, 3, stride=2, padding=1, bias=True)
        self.r2 = nn.Conv2d(nf*4, nf*8, 3, stride=2, padding=1, bias=True)
        self.r3 = nn.Conv2d(nf*8, nf*16, 3, stride=2, padding=1, bias=True)
        self.reducers = [self.r1, self.r2, self.r3]

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.a1 = nn.Conv2d(nf*4, nf*8, 3, stride=1, padding=1, bias=True)
        self.a2 = nn.Conv2d(nf*2, nf*4, 3, stride=1, padding=1, bias=True)
        self.a3 = nn.Conv2d(nf, nf, 3, stride=1, padding=1, bias=True)
        self.assemblers = [self.a1, self.a2, self.a3]

        if self.downscale == 1:
            nf_last = nf
        elif self.downscale == 2:
            nf_last = nf * 4
        elif self.downscale == 4:
            nf_last = nf * 8

        self.conv_last = nn.Conv2d(nf_last, out_nc, 3, stride=1, padding=1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.conv_last] + self.reducers + self.assemblers,
                                     .1)

    def forward(self, x):
        # Noise has the same shape as the input with only one channel.
        rand_feature = torch.randn((x.shape[0], 1) + x.shape[2:], device=x.device, dtype=x.dtype)
        out = torch.cat([x, rand_feature], dim=1)

        out = self.lrelu(self.conv_first(out))
        skips = []
        for i in range(4):
            skips.append(out)
            out = self.trunks[i](out)
            if i < 3:
                out = self.lrelu(self.reducers[i](out))

        target_width = x.shape[-1] / self.downscale
        i = 0
        while out.shape[-1] != target_width:
            out = self.pixel_shuffle(out)
            out = self.lrelu(self.assemblers[i](out))
            out = out + skips[-i-2]
            i += 1

        # TODO: Figure out where this magic number '12' comes from and fix it.
        out = 12 * self.conv_last(out)
        if self.downscale == 1:
            base = x
        else:
            base = F.interpolate(x, scale_factor=1/self.downscale, mode='bilinear', align_corners=False)
        return out + base
