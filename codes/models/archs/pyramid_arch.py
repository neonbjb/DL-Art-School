import torch
from torch import nn

from models.archs.arch_util import ConvGnLelu, ExpansionBlock
from models.flownet2.networks.resample2d_package.resample2d import Resample2d
from utils.util import checkpoint
import torch.nn.functional as F


class Pyramid(nn.Module):
    def __init__(self, nf, depth, processing_convs_per_layer, processing_at_point, scale_per_level=2, block=ConvGnLelu,
                 norm=True, return_outlevels=False):
        super(Pyramid, self).__init__()
        levels = []
        current_filters = nf
        self.return_outlevels = return_outlevels
        for d in range(depth):
            level = [block(current_filters, int(current_filters*scale_per_level), kernel_size=3, stride=2, activation=True, norm=False, bias=False)]
            current_filters = int(current_filters*scale_per_level)
            for pc in range(processing_convs_per_layer):
                level.append(block(current_filters, current_filters, kernel_size=3, activation=True, norm=norm, bias=False))
            levels.append(nn.Sequential(*level))
        self.downsamples = nn.ModuleList(levels)
        if processing_at_point > 0:
            point_processor = []
            for p in range(processing_at_point):
                point_processor.append(block(current_filters, current_filters, kernel_size=3, activation=True, norm=norm, bias=False))
            self.point_processor = nn.Sequential(*point_processor)
        else:
            self.point_processor = None
        levels = []
        for d in range(depth):
            level = [ExpansionBlock(current_filters, int(current_filters / scale_per_level), block=block)]
            current_filters = int(current_filters / scale_per_level)
            for pc in range(processing_convs_per_layer):
                level.append(block(current_filters, current_filters, kernel_size=3, activation=True, norm=norm, bias=False))
            levels.append(nn.ModuleList(level))
        self.upsamples = nn.ModuleList(levels)

    def forward(self, x):
        passthroughs = []
        fea = x
        for lvl in self.downsamples:
            passthroughs.append(fea)
            fea = lvl(fea)
        out_levels = []
        fea = self.point_processor(fea)
        for i, lvl in enumerate(self.upsamples):
            out_levels.append(fea)
            for j, sublvl in enumerate(lvl):
                if j == 0:
                    fea = sublvl(fea, passthroughs[-1-i])
                else:
                    fea = sublvl(fea)

        out_levels.append(fea)

        if self.return_outlevels:
            return tuple(out_levels)
        else:
            return fea


class BasicResamplingFlowNet(nn.Module):
    def create_termini(self, filters):
        return nn.Sequential(ConvGnLelu(int(filters), 2, kernel_size=3, activation=False, norm=False, bias=True),
                             nn.Tanh())

    def __init__(self, nf, resample_scale=1):
        super(BasicResamplingFlowNet, self).__init__()
        self.initial_conv = ConvGnLelu(6, nf, kernel_size=7, activation=False, norm=False, bias=True)
        self.pyramid = Pyramid(nf, 3, 0, 1, 1.5, return_outlevels=True)
        self.termini = nn.ModuleList([self.create_termini(nf*1.5**3),
                                      self.create_termini(nf*1.5**2),
                                      self.create_termini(nf*1.5)])
        self.terminus = nn.Sequential(ConvGnLelu(nf, nf, kernel_size=3, activation=True, norm=True, bias=True),
                                      ConvGnLelu(nf, nf, kernel_size=3, activation=True, norm=True, bias=False),
                                      ConvGnLelu(nf, nf//2, kernel_size=3, activation=False, norm=False, bias=True),
                                      ConvGnLelu(nf//2, 2, kernel_size=3, activation=False, norm=False, bias=True),
                                      nn.Tanh())
        self.scale = resample_scale
        self.resampler = Resample2d()

    def forward(self, left, right):
        fea = self.initial_conv(torch.cat([left, right], dim=1))
        levels = checkpoint(self.pyramid, fea)
        flos = []
        compares = []
        for i, level in enumerate(levels):
            if i == 3:
                flow = checkpoint(self.terminus, level) * self.scale
            else:
                flow = self.termini[i](level) * self.scale
            img_scale = 1/2**(3-i)
            flos.append(self.resampler(F.interpolate(left, scale_factor=img_scale, mode="area").float(), flow.float()))
            compares.append(F.interpolate(right, scale_factor=img_scale, mode="area"))
        flos_structural_var = torch.var(flos[-1], dim=[-1,-2])
        return flos, compares, flos_structural_var
