from typing import Optional, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.utils import _ntuple
import torch.nn.functional as F


_pair = _ntuple(2)


# Indexes the <p> index of input=b,c,h,w,p by the long tensor index=b,1,h,w. Result is b,c,h,w.
# Frankly - IMO - this is what torch.gather should do.
def index_2d(input, index):
    index = index.repeat(1,input.shape[1],1,1)
    e = torch.eye(input.shape[-1], device=input.device)
    result = e[index] * input
    return result.sum(-1)


# Drop-in implementation of Conv2d that can apply masked scales&shifts to the convolution weights.
class ScaledWeightConv(_ConvNd):
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
        breadth: int = 8,
    ):
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, _pair(kernel_size), stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.weight_scales = nn.ParameterList([nn.Parameter(torch.ones(out_channels, in_channels, kernel_size, kernel_size)) for _ in range(breadth)])
        self.shifts = nn.ParameterList([nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size)) for _ in range(breadth)])
        for w, s in zip(self.weight_scales, self.shifts):
            w.FOR_SCALE_SHIFT = True
            s.FOR_SCALE_SHIFT = True
        # This should probably be configurable at some point.
        self.weight.DO_NOT_TRAIN = True
        self.weight.requires_grad = False

    def _weighted_conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor, masks: dict = None) -> Tensor:
        if masks is None:
            # An alternate "mode" of operation is the masks are injected as parameters.
            assert hasattr(self, 'masks')
            masks = self.masks

        # This is an exceptionally inefficient way of achieving this functionality. The hope is that if this is any
        # good at all, this can be made more efficient by performing a single conv pass with multiple masks.
        weighted_convs = [self._weighted_conv_forward(input, self.weight * scale + shift) for scale, shift in zip(self.weight_scales, self.shifts)]
        weighted_convs = torch.stack(weighted_convs, dim=-1)

        needed_mask = weighted_convs.shape[-2]
        assert needed_mask in masks.keys()

        return index_2d(weighted_convs, masks[needed_mask])


def create_wrapped_conv_from_template(conv: nn.Conv2d, breadth: int):
    wrapped = ScaledWeightConv(conv.in_channels,
                               conv.out_channels,
                               conv.kernel_size[0],
                               conv.stride[0],
                               conv.padding[0],
                               conv.dilation[0],
                               conv.groups,
                               conv.bias,
                               conv.padding_mode,
                               breadth)
    return wrapped


# Drop-in implementation of ConvTranspose2d that can apply masked scales&shifts to the convolution weights.
class ScaledWeightConvTranspose(_ConvTransposeNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = 'zeros',
        breadth: int = 8,
    ):
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super().__init__(
            in_channels, out_channels, _pair(kernel_size), stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode)

        self.weight_scales = nn.ParameterList([nn.Parameter(torch.ones(in_channels, out_channels, kernel_size, kernel_size)) for _ in range(breadth)])
        self.shifts = nn.ParameterList([nn.Parameter(torch.zeros(in_channels, out_channels, kernel_size, kernel_size)) for _ in range(breadth)])
        for w, s in zip(self.weight_scales, self.shifts):
            w.FOR_SCALE_SHIFT = True
            s.FOR_SCALE_SHIFT = True
        # This should probably be configurable at some point.
        self.weight.DO_NOT_TRAIN = True
        self.weight.requires_grad = False

    def _conv_transpose_forward(self, input, weight, output_size) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)

        return F.conv_transpose2d(
            input, weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

    def forward(self, input: Tensor, masks: dict = None, output_size: Optional[List[int]] = None) -> Tensor:
        if masks is None:
            # An alternate "mode" of operation is the masks are injected as parameters.
            assert hasattr(self, 'masks')
            masks = self.masks

        # This is an exceptionally inefficient way of achieving this functionality. The hope is that if this is any
        # good at all, this can be made more efficient by performing a single conv pass with multiple masks.
        weighted_convs = [self._conv_transpose_forward(input, self.weight * scale + shift, output_size)
                          for scale, shift in zip(self.weight_scales, self.shifts)]
        weighted_convs = torch.stack(weighted_convs, dim=-1)

        needed_mask = weighted_convs.shape[-2]
        assert needed_mask in masks.keys()

        return index_2d(weighted_convs, masks[needed_mask])


def create_wrapped_conv_transpose_from_template(conv: nn.Conv2d, breadth: int):
    wrapped = ScaledWeightConvTranspose(conv.in_channels,
                               conv.out_channels,
                               conv.kernel_size,
                               conv.stride,
                               conv.padding,
                               conv.output_padding,
                               conv.groups,
                               conv.bias,
                               conv.dilation,
                               conv.padding_mode,
                               breadth)
    wrapped.weight = conv.weight
    wrapped.weight.DO_NOT_TRAIN = True
    wrapped.weight.requires_grad = False
    wrapped.bias = conv.bias
    return wrapped
