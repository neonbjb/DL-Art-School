import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from math import sqrt

def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(torch.pow(x, 2), dim=1, keepdims=True) + epsilon)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers, return_layers=False):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    if return_layers:
        return nn.Sequential(*layers), layers
    else:
        return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    '''Residual block with BN
    ---Conv-BN-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.BN1 = nn.BatchNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.BN2 = nn.BatchNorm2d(nf)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.BN1(self.conv1(x)))
        out = self.BN2(self.conv2(out))
        return identity + out

class ResidualBlockSpectralNorm(nn.Module):
    '''Residual block with Spectral Normalization.
    ---SpecConv-ReLU-SpecConv-+-
     |________________|
    '''

    def __init__(self, nf, total_residual_blocks):
        super(ResidualBlockSpectralNorm, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = SpectralNorm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        self.conv2 = SpectralNorm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

        initialize_weights([self.conv1, self.conv2], 1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return identity + out

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output


class PixelUnshuffle(nn.Module):
    def __init__(self, reduction_factor):
        super(PixelUnshuffle, self).__init__()
        self.r = reduction_factor

    def forward(self, x):
        (b, f, w, h) = x.shape
        x = x.contiguous().view(b, f, w // self.r, self.r, h // self.r, self.r)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, f * (self.r ** 2), w // self.r, h // self.r)
        return x

