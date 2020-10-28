import torch
import torch.nn as nn
from utils.util import sequential_checkpoint
from models.archs.arch_util import ConvGnSilu, make_layer


class TecoResblock(nn.Module):
    def __init__(self, nf):
        super(TecoResblock, self).__init__()
        self.nf = nf
        self.conv1 = ConvGnSilu(nf, nf, kernel_size=3, norm=False, activation=True, bias=False, weight_init_factor=.1)
        self.conv2 = ConvGnSilu(nf, nf, kernel_size=3, norm=False, activation=False, bias=False, weight_init_factor=.1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        return identity + x


class TecoUpconv(nn.Module):
    def __init__(self, nf, scale):
        super(TecoUpconv, self).__init__()
        self.nf = nf
        self.scale = scale
        self.conv1 = ConvGnSilu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.conv2 = ConvGnSilu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.conv3 = ConvGnSilu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.final_conv = ConvGnSilu(nf, 3, kernel_size=1, norm=False, activation=False, bias=False)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")
        x = self.conv3(x)
        return self.final_conv(x)


# Extremely simple resnet based generator that is very similar to the one used in the tecogan paper.
# Main differences:
# - Uses SiLU instead of ReLU
# - Reference input is in HR space (just makes more sense)
# - Doesn't use transposed convolutions - just uses interpolation instead.
# - Upsample block is slightly more complicated.
class TecoGen(nn.Module):
    def __init__(self, nf, scale):
        super(TecoGen, self).__init__()
        self.nf = nf
        self.scale = scale
        fea_conv = ConvGnSilu(6, nf, kernel_size=7, stride=self.scale, bias=True, norm=False, activation=True)
        res_layers = [TecoResblock(nf) for i in range(15)]
        upsample = TecoUpconv(nf, scale)
        everything = [fea_conv] + res_layers + [upsample]
        self.core = nn.Sequential(*everything)

    def forward(self, x, ref=None):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode="bicubic")
        if ref is None:
            ref = torch.zeros_like(x)
        join = torch.cat([x, ref], dim=1)
        return x + sequential_checkpoint(self.core, 6, join)

