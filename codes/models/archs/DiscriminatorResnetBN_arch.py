import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.lrelu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_filters=16, num_classes=10):
        super(ResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = num_filters
        self.conv1 = conv3x3(3, num_filters)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.layer1 = self._make_layer(block, num_filters, layers[0])
        self.layer2 = self._make_layer(block, num_filters * 2, layers[1], stride=2)
        self.skip_conv1 = conv3x3(3, num_filters*2)
        self.layer3 = self._make_layer(block, num_filters * 4, layers[2], stride=2)
        self.skip_conv2 = conv3x3(3, num_filters*4)
        self.layer4 = self._make_layer(block, num_filters * 8, layers[2], stride=2)
        self.fc1 = nn.Linear(num_filters * 8 * 8 * 8, 64, bias=True)
        self.fc2 = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.BatchNorm2d(self.inplanes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, gen_skips=None):
        x_dim = x.size(-1)
        if gen_skips is None:
            gen_skips = {
                int(x_dim/2): F.interpolate(x, scale_factor=1/2, mode='bilinear', align_corners=False),
                int(x_dim/4): F.interpolate(x, scale_factor=1/4, mode='bilinear', align_corners=False),
            }
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = (x + self.skip_conv1(gen_skips[int(x_dim/2)])) / 2
        x = self.layer3(x)
        x = (x + self.skip_conv2(gen_skips[int(x_dim/4)])) / 2
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.lrelu(self.fc1(x))
        x = self.fc2(x)

        return x


def resnet20(**kwargs):
    """Constructs a ResNet-20 model.
    """
    model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32(**kwargs):
    """Constructs a ResNet-32 model.
    """
    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44(**kwargs):
    """Constructs a ResNet-44 model.
    """
    model = ResNet(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56(**kwargs):
    """Constructs a ResNet-56 model.
    """
    model = ResNet(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110(**kwargs):
    """Constructs a ResNet-110 model.
    """
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202(**kwargs):
    """Constructs a ResNet-1202 model.
    """
    model = ResNet(BasicBlock, [200, 200, 200], **kwargs)
    return model