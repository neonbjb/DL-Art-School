import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


__all__ = ['FixupResNet', 'fixup_resnet18', 'fixup_resnet34', 'fixup_resnet50', 'fixup_resnet101', 'fixup_resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, conv_create=conv3x3):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv_create(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv_create(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.lrelu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)

        out += identity
        out = self.lrelu(out)

        return out


class FixupResNet(nn.Module):

    def __init__(self, block, layers, upscale_applications=2, num_filters=64, inject_noise=False):
        super(FixupResNet, self).__init__()
        self.inject_noise = inject_noise
        self.num_layers = sum(layers) + layers[-1] * (upscale_applications - 1)  # The last layer is applied repeatedly to achieve high level SR.
        self.inplanes = num_filters
        self.upscale_applications = upscale_applications
        # Part 1 - Process raw input image. Most denoising should appear here and this should be the most complicated
        # part of the block.
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.layer1 = self._make_layer(block, num_filters, layers[0], stride=1)
        self.skip1 = nn.Conv2d(num_filters, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.skip1_bias = nn.Parameter(torch.zeros(1))

        # Part 2 - This is the upsampler core. It consists of a normal multiplicative conv followed by several residual
        #          convs which are intended to repair artifacts caused by 2x interpolation.
        #          This core layer should by itself accomplish 2x super-resolution. We use it in repeat to do the
        #          requested SR.
        self.nf2 = int(num_filters/4)
        # This part isn't repeated. It de-filters the output from the previous step to fit the filter size used in the
        # upsampler-conv.
        self.upsampler_conv = nn.Conv2d(num_filters, self.nf2, kernel_size=3, stride=1, padding=1, bias=False)
        self.uc_bias = nn.Parameter(torch.zeros(1))
        self.inplanes = self.nf2

        if layers[1] > 0:
            # This is the repeated part.
            self.layer2 = self._make_layer(block, int(self.nf2), layers[1], stride=1, conv_type=conv5x5)
            self.skip2 = nn.Conv2d(self.nf2, 3, kernel_size=5, stride=1, padding=2, bias=False)
            self.skip2_bias = nn.Parameter(torch.zeros(1))

        self.final_defilter = nn.Conv2d(self.nf2, 3, kernel_size=5, stride=1, padding=2, bias=True)
        self.bias2 = nn.Parameter(torch.zeros(1))

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))


    def _make_layer(self, block, planes, blocks, stride=1, conv_type=conv3x3):
        defilter = None
        if self.inplanes != planes * block.expansion:
            defilter = conv1x1(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, defilter))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_create=conv_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.inject_noise:
            rand_feature = torch.randn_like(x)
            x = x + rand_feature * .1
        x = self.conv1(x)
        x = self.lrelu(x + self.bias1)
        x = self.layer1(x)
        skip_lo = self.skip1(x) + self.skip1_bias

        x = self.lrelu(self.upsampler_conv(x) + self.uc_bias)
        if self.upscale_applications > 0:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.layer2(x)
            skip_med = self.skip2(x) + self.skip2_bias
        else:
            skip_med = skip_lo

        if self.upscale_applications > 1:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.layer2(x)

        x = self.final_defilter(x) + self.bias2
        return x, skip_med, skip_lo

class FixupResNetV2(FixupResNet):
    def __init__(self, **kwargs):
        super(FixupResNetV2, self).__init__(**kwargs)
        # Use one unified filter-to-image stack, not the previous skip stacks.
        self.skip1 = None
        self.skip1_bias = None
        self.skip2 = None
        self.skip2_bias = None
        # The new filter-to-image stack will be 2 conv layers deep, not 1.
        self.final_process = nn.Conv2d(self.nf2, self.nf2, kernel_size=5, stride=1, padding=2, bias=True)
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fp_bn = nn.BatchNorm2d(self.nf2)
        self.final_defilter = nn.Conv2d(self.nf2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.bias3 = nn.Parameter(torch.zeros(1))

    def filter_to_image(self, filter):
        x = self.final_process(filter) + self.bias2
        x = self.lrelu(self.fp_bn(x))
        x = self.final_defilter(x) + self.bias3
        return x

    def forward(self, x):
        if self.inject_noise:
            rand_feature = torch.randn_like(x)
            x = x + rand_feature * .1
        x = self.conv1(x)
        x = self.lrelu(x + self.bias1)
        x = self.layer1(x)
        x = self.lrelu(self.upsampler_conv(x) + self.uc_bias)

        skip_lo = self.filter_to_image(x)
        if self.upscale_applications > 0:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.layer2(x)

        skip_med = self.filter_to_image(x)
        if self.upscale_applications > 1:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.layer2(x)

        if self.upscale_applications == 2:
            x = self.filter_to_image(x)
        elif self.upscale_applications == 1:
            x = skip_med
            skip_med = skip_lo
            skip_lo = None
        elif self.upscale_applications == 0:
            x = skip_lo
            skip_lo = None
            skip_med = None

        return x, skip_med, skip_lo

def fixup_resnet34(nb_denoiser=20, nb_upsampler=10, **kwargs):
    """Constructs a Fixup-ResNet-34 model.
    """
    model = FixupResNet(FixupBasicBlock, [nb_denoiser, nb_upsampler], **kwargs)
    return model

def fixup_resnet34_v2(nb_denoiser=20, nb_upsampler=10, **kwargs):
    """Constructs a Fixup-ResNet-34 model.
    """
    kwargs['block'] = FixupBasicBlock
    kwargs['layers'] = [nb_denoiser, nb_upsampler]
    model = FixupResNetV2(**kwargs)
    return model


__all__ = ['FixupResNet', 'fixup_resnet34', 'fixup_resnet34_v2']