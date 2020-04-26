import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch


class HighToLowResNet(nn.Module):
    ''' ResNet that applies a noise channel to the input, then downsamples it. Currently only downscale=4 is supported. '''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, downscale=4):
        super(HighToLowResNet, self).__init__()
        self.downscale = downscale

        # We will always apply a noise channel to the inputs, account for that here.
        in_nc += 1

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        basic_block2 = functools.partial(arch_util.ResidualBlock_noBN, nf=nf*2)
        # To keep the total model size down, the residual trunks will be applied across 3 downsampling stages.
        # The first will be applied against the hi-res inputs and will have only 4 layers.
        # The second will be applied after half of the downscaling and will also have only 6 layers.
        # The final will be applied against the final resolution and will have all of the remaining layers.
        self.trunk_hires = arch_util.make_layer(basic_block, 5)
        self.trunk_medres = arch_util.make_layer(basic_block, 10)
        self.trunk_lores = arch_util.make_layer(basic_block2, nb - 15)

        # downsampling
        if self.downscale == 4 or self.downscale == 1:
            self.downconv1 = nn.Conv2d(nf, nf, 3, stride=2, padding=1, bias=True)
            self.downconv2 = nn.Conv2d(nf, nf*2, 3, stride=2, padding=1, bias=True)
        else:
            raise EnvironmentError("Requested downscale not supported: %i" % (downscale,))

        self.HRconv = nn.Conv2d(nf*2, nf*2, 3, stride=1, padding=1, bias=True)
        if self.downscale == 4:
            self.conv_last = nn.Conv2d(nf*2, out_nc, 3, stride=1, padding=1, bias=True)
        else:
            self.pixel_shuffle = nn.PixelShuffle(4)
            self.conv_last = nn.Conv2d(int(nf/8), out_nc, 3, stride=1, padding=1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.HRconv, self.conv_last, self.downconv1, self.downconv2],
                                     0.1)

    def forward(self, x):
        # Noise has the same shape as the input with only one channel.
        rand_feature = torch.randn((x.shape[0], 1) + x.shape[2:], device=x.device)
        out = torch.cat([x, rand_feature], dim=1)

        out = self.lrelu(self.conv_first(out))
        out = self.trunk_hires(out)

        if self.downscale == 4 or self.downscale == 1:
            out = self.lrelu(self.downconv1(out))
            out = self.trunk_medres(out)
            out = self.lrelu(self.downconv2(out))
            out = self.trunk_lores(out)

        if self.downscale == 1:
            out = self.lrelu(self.pixel_shuffle(self.HRconv(out)))
            out = self.conv_last(out)
        else:
            out = self.conv_last(self.lrelu(self.HRconv(out)))

        if self.downscale == 1:
            base = x
        else:
            base = F.interpolate(x, scale_factor=1/self.downscale, mode='bilinear', align_corners=False)

        out += base
        return out
