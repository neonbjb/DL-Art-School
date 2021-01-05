import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, init
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple

_pair = _ntuple(2)

class TransferConv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        transfer_mode: bool = False
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.transfer_mode = transfer_mode
        if transfer_mode:
            self.transfer_scale = nn.Parameter(torch.ones(out_channels, in_channels, 1, 1))
            self.transfer_scale.FOR_TRANSFER_LEARNING = True
            self.transfer_shift = nn.Parameter(torch.zeros(out_channels, in_channels, 1, 1))
            self.transfer_shift.FOR_TRANSFER_LEARNING = True

    def _conv_forward(self, input, weight):
        if self.transfer_mode:
            weight = weight * self.transfer_scale + self.transfer_shift
        else:
            weight = weight

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight)


class TransferLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, transfer_mode: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.transfer_mode = transfer_mode
        if transfer_mode:
            self.transfer_scale = nn.Parameter(torch.ones(out_features, in_features))
            self.transfer_scale.FOR_TRANSFER_LEARNING = True
            self.transfer_shift = nn.Parameter(torch.zeros(out_features, in_features))
            self.transfer_shift.FOR_TRANSFER_LEARNING = True

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if self.transfer_mode:
            weight = self.weight * self.transfer_scale + self.transfer_shift
        else:
            weight = self.weight
        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class TransferConvGnLelu(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3, stride=1, activation=True, norm=True, bias=True, num_groups=8, weight_init_factor=1, transfer_mode=False):
        super().__init__()
        padding_map = {1: 0, 3: 1, 5: 2, 7: 3}
        assert kernel_size in padding_map.keys()
        self.conv = TransferConv2d(filters_in, filters_out, kernel_size, stride, padding_map[kernel_size], bias=bias, transfer_mode=transfer_mode)
        if norm:
            self.gn = nn.GroupNorm(num_groups, filters_out)
        else:
            self.gn = None
        if activation:
            self.lelu = nn.LeakyReLU(negative_slope=.2)
        else:
            self.lelu = None

        # Init params.
        for m in self.modules():
            if isinstance(m, TransferConv2d):
                nn.init.kaiming_normal_(m.weight, a=.1, mode='fan_out',
                                        nonlinearity='leaky_relu' if self.lelu else 'linear')
                m.weight.data *= weight_init_factor
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.gn:
            x = self.gn(x)
        if self.lelu:
            return self.lelu(x)
        else:
            return x