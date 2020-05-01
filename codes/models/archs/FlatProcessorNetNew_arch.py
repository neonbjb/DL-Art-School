import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
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

    def __init__(self, block, num_filters, layers, num_classes=1000):
        super(FixupResNet, self).__init__()
        self.num_layers = sum(layers)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # 4 input channels, including the noise.
        self.conv1 = nn.Conv2d(4, num_filters, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.inplanes = num_filters
        self.down_layer1 = self._make_layer(block, num_filters, layers[0])
        self.down_layer2 = self._make_layer(block, num_filters, layers[1], stride=2)
        self.down_layer3 = self._make_layer(block, num_filters * 4, layers[2], stride=2)
        self.down_layer4 = self._make_layer(block, num_filters * 16, layers[3], stride=2)

        self.inplanes = num_filters * 4
        self.up_layer1 = self._make_layer(block, num_filters * 4, layers[4], stride=1)
        self.inplanes = num_filters
        self.up_layer2 = self._make_layer(block, num_filters, layers[5], stride=1)

        self.defilter = nn.Conv2d(num_filters, 3, kernel_size=5, stride=1, padding=2, bias=False)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        skip = x

        # Noise has the same shape as the input with only one channel.
        rand_feature = torch.randn((x.shape[0], 1) + x.shape[2:], device=x.device, dtype=x.dtype)
        x = torch.cat([x, rand_feature], dim=1)

        x = self.conv1(x)
        x = self.lrelu(x + self.bias1)

        x = self.down_layer1(x)
        x = self.down_layer2(x)
        x = self.down_layer3(x)
        x = self.down_layer4(x)

        x = self.pixel_shuffle(x)
        x = self.up_layer1(x)
        x = self.pixel_shuffle(x)
        x = self.up_layer2(x)

        x = self.defilter(x)

        base = F.interpolate(skip, scale_factor=.25, mode='bilinear', align_corners=False)
        return x + base


def fixup_resnet34(num_filters, **kwargs):
    """Constructs a Fixup-ResNet-34 model.
    """
    model = FixupResNet(FixupBasicBlock, num_filters, [3, 4, 6, 3, 2, 2], **kwargs)
    return model