import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch

class ReduceAnnealer(nn.Module):
    '''
    Reduces an image dimensionality by half and performs a specified number of residual blocks on it before
    `annealing` the filter count to the same as the input filter count.

    To reduce depth, accepts an interpolated "trunk" input which is summed with the output of the RA block before
    returning.

    Returns a tuple in the forward pass. The first return is the annealed output. The second is the output before
    annealing (e.g. number_filters=input*4) which can be be used for upsampling.
    '''

    def __init__(self, number_filters, residual_blocks):
        super(ReduceAnnealer, self).__init__()
        self.reducer = nn.Conv2d(number_filters, number_filters*4, 3, stride=2, padding=1, bias=True)
        self.res_trunk = arch_util.make_layer(functools.partial(arch_util.ResidualBlock, nf=number_filters*4), residual_blocks)
        self.annealer = nn.Conv2d(number_filters*4, number_filters, 3, stride=1, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        arch_util.initialize_weights([self.reducer, self.annealer], .1)

    def forward(self, x, interpolated_trunk):
        out = self.lrelu(self.reducer(x))
        out = self.lrelu(self.res_trunk(out))
        annealed = self.lrelu(self.annealer(out)) + interpolated_trunk
        return annealed, out

class Assembler(nn.Module):
    '''
    Upsamples a given input using PixelShuffle. Then upsamples this input further and adds in a residual raw input from
    a corresponding upstream ReduceAnnealer. Finally performs processing using ResNet blocks.
    '''
    def __init__(self, number_filters, residual_blocks):
        super(Assembler, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsampler = nn.Conv2d(number_filters, number_filters*4, 3, stride=1, padding=1, bias=True)
        self.res_trunk = arch_util.make_layer(functools.partial(arch_util.ResidualBlock, nf=number_filters*4), residual_blocks)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input, skip_raw):
        out = self.pixel_shuffle(input)
        out = self.upsampler(out) + skip_raw
        out = self.lrelu(self.res_trunk(out))
        return out

class FlatProcessorNet(nn.Module):
    '''
    Specialized network that tries to perform a near-equal amount of processing on each of 5 downsampling steps. Image
    is then upsampled to a specified size with a similarly flat amount of processing.

    This network automatically applies a noise vector on the inputs to provide entropy for processing.
     '''
    def __init__(self, in_nc=3, out_nc=3, nf=64, reduce_anneal_blocks=4, assembler_blocks=2, downscale=4):
        super(FlatProcessorNet, self).__init__()

        assert downscale in [1, 2, 4], "Requested downscale not supported; %i" % (downscale, )
        self.downscale = downscale

        # We will always apply a noise channel to the inputs, account for that here.
        in_nc += 1

        # We need two layers to move the image into the filter space in which we will perform most of the work.
        self.conv_first = nn.Conv2d(in_nc, nf, 3, stride=1, padding=1, bias=True)
        self.conv_last = nn.Conv2d(nf*4, out_nc, 3, stride=1, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Torch modules need to have all submodules as explicit class members. So make those, then add them into an
        # array for easier logic in forward().
        self.ra1 = ReduceAnnealer(nf, reduce_anneal_blocks)
        self.ra2 = ReduceAnnealer(nf, reduce_anneal_blocks)
        self.ra3 = ReduceAnnealer(nf, reduce_anneal_blocks)
        self.ra4 = ReduceAnnealer(nf, reduce_anneal_blocks)
        self.ra5 = ReduceAnnealer(nf, reduce_anneal_blocks)
        self.reducers = [self.ra1, self.ra2, self.ra3, self.ra4, self.ra5]

        # Produce assemblers for all possible downscale variants. Some may not be used.
        self.assembler1 = Assembler(nf, assembler_blocks)
        self.assembler2 = Assembler(nf, assembler_blocks)
        self.assembler3 = Assembler(nf, assembler_blocks)
        self.assembler4 = Assembler(nf, assembler_blocks)
        self.assemblers = [self.assembler1, self.assembler2, self.assembler3, self.assembler4]

        # Initialization
        arch_util.initialize_weights([self.conv_first, self.conv_last], .1)

    def forward(self, x):
        # Noise has the same shape as the input with only one channel.
        rand_feature = torch.randn((x.shape[0], 1) + x.shape[2:], device=x.device, dtype=x.dtype)
        out = torch.cat([x, rand_feature], dim=1)

        out = self.lrelu(self.conv_first(out))
        features_trunk = out
        raw_values = []
        downsamples = 1
        for ra in self.reducers:
            downsamples *= 2
            interpolated = F.interpolate(features_trunk, scale_factor=1/downsamples, mode='bilinear', align_corners=False)
            out, raw = ra(out, interpolated)
            raw_values.append(raw)

        i = -1
        out = raw_values[-1]
        while downsamples != self.downscale:
            out = self.assemblers[i](out, raw_values[i-1])
            i -= 1
            downsamples = int(downsamples / 2)

        out = self.conv_last(out)

        basis = x
        if downsamples != 1:
            basis = F.interpolate(x, scale_factor=1/downsamples, mode='bilinear', align_corners=False)
        return basis + out
