import torch
import torch.nn as nn
import torchvision
import models.archs.arch_util as arch_util
import functools
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm

# Class that halfs the image size (x4 complexity reduction) and doubles the filter size. Substantial resnet
# processing is also performed.
class ResnetDownsampleLayer(nn.Module):
    def __init__(self, starting_channels: int, number_filters: int, filter_multiplier: int, residual_blocks_input: int, residual_blocks_skip_image: int, total_residual_blocks: int):
        super(ResnetDownsampleLayer, self).__init__()

        self.skip_image_reducer = SpectralNorm(nn.Conv2d(starting_channels, number_filters, 3, stride=1, padding=1, bias=True))
        self.skip_image_res_trunk = arch_util.make_layer(functools.partial(arch_util.ResidualBlockSpectralNorm, nf=number_filters, total_residual_blocks=total_residual_blocks), residual_blocks_skip_image)

        self.input_reducer = SpectralNorm(nn.Conv2d(number_filters, number_filters*filter_multiplier, 3, stride=2, padding=1, bias=True))
        self.res_trunk = arch_util.make_layer(functools.partial(arch_util.ResidualBlockSpectralNorm, nf=number_filters*filter_multiplier, total_residual_blocks=total_residual_blocks), residual_blocks_input)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        arch_util.initialize_weights([self.input_reducer, self.skip_image_reducer], 1)

    def forward(self, x, skip_image):
        # Process the skip image first.
        skip = self.lrelu(self.skip_image_reducer(skip_image))
        skip = self.skip_image_res_trunk(skip)

        # Concat the processed skip image onto the input and perform processing.
        out = (x + skip) / 2
        out = self.lrelu(self.input_reducer(out))
        out = self.res_trunk(out)
        return out

class DiscriminatorResnet(nn.Module):
    # Discriminator that downsamples 5 times with resnet blocks at each layer. On each downsample, the filter size is
    # increased by a factor of 2. Feeds the output of the convs into a dense for prediction at the logits. Scales the
    # final dense based on the input image size. Intended for use with input images which are multiples of 32.
    #
    # This discriminator also includes provisions to pass an image at various downsample steps in directly. When this
    # is done with a generator, it will allow much shorter gradient paths between the generator and discriminator. When
    # no downsampled images are passed into the forward() pass, they will be automatically generated from the source
    # image using interpolation.
    #
    # Uses spectral normalization rather than batch normalization.
    def __init__(self, in_nc: int, nf: int, input_img_size: int, trunk_resblocks: int, skip_resblocks: int):
        super(DiscriminatorResnet, self).__init__()
        self.dimensionalize = nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, padding=1, bias=True)

        # Trunk resblocks are the important things to get right, so use those. 5=number of downsample layers.
        total_resblocks = trunk_resblocks * 5
        self.downsample1 = ResnetDownsampleLayer(in_nc, nf, 2, trunk_resblocks, skip_resblocks, total_resblocks)
        self.downsample2 = ResnetDownsampleLayer(in_nc, nf*2, 2, trunk_resblocks, skip_resblocks, total_resblocks)
        self.downsample3 = ResnetDownsampleLayer(in_nc, nf*4, 2, trunk_resblocks, skip_resblocks, total_resblocks)
        # At the bottom layers, we cap the filter multiplier. We want this particular network to focus as much on the
        # macro-details at higher image dimensionality as it does to the feature details.
        self.downsample4 = ResnetDownsampleLayer(in_nc, nf*8, 1, trunk_resblocks, skip_resblocks, total_resblocks)
        self.downsample5 = ResnetDownsampleLayer(in_nc, nf*8, 1, trunk_resblocks, skip_resblocks, total_resblocks)
        self.downsamplers = [self.downsample1, self.downsample2, self.downsample3, self.downsample4, self.downsample5]

        downsampled_image_size = input_img_size / 32
        self.linear1 = nn.Linear(int(nf * 8 * downsampled_image_size * downsampled_image_size), 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        arch_util.initialize_weights([self.dimensionalize, self.linear1, self.linear2], 1)

    def forward(self, x, skip_images=None):
        if skip_images is None:
            # Sythesize them from x.
            skip_images = []
            for i in range(len(self.downsamplers)):
                m = 2 ** i
                skip_images.append(F.interpolate(x, scale_factor=1 / m, mode='bilinear', align_corners=False))

        fea = self.dimensionalize(x)
        for skip, d in zip(skip_images, self.downsamplers):
            fea = d(fea, skip)

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out
