import functools
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init, Conv2d
import torch.nn.functional as F


class SwitchedConv(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        switch_breadth: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        include_coupler: bool = False,  # A 'coupler' is a latent converter which can make any bxcxhxw tensor a compatible switchedconv selector by performing a linear 1x1 conv, softmax and interpolate.
        coupler_dim_in: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.groups = groups

        if include_coupler:
            self.coupler = Conv2d(coupler_dim_in, switch_breadth, kernel_size=1)
        else:
            self.coupler = None

        self.weights = nn.ParameterList([nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)) for _ in range(switch_breadth)])
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w in self.weights:
            init.kaiming_uniform_(w, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights[0])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inp, selector):
        if self.coupler:
            selector = F.softmax(self.coupler(selector), dim=1)
            out_shape = [s // self.stride for s in inp.shape[2:]]
            if selector.shape[2] != out_shape[0] or selector.shape[3] != out_shape[1]:
                selector = F.interpolate(selector, size=out_shape, mode="nearest")

        conv_results = []
        for i, w in enumerate(self.weights):
            conv_results.append(F.conv2d(inp, w, self.bias, self.stride, self.padding, self.dilation, self.groups) * selector[:, i].unsqueeze(1))
        return torch.stack(conv_results, dim=-1).sum(dim=-1)



# Given a state_dict and the module that that sd belongs to, strips out all Conv2d.weight parameters and replaces them
# with the equivalent SwitchedConv.weight parameters. Does not create coupler params.
def convert_conv_net_state_dict_to_switched_conv(module, switch_breadth, ignore_list=[]):
    state_dict = module.state_dict()
    for name, m in module.named_modules():
        ignored = False
        for smod in ignore_list:
            if smod in name:
                ignored = True
                continue
        if ignored:
            continue
        if isinstance(m, nn.Conv2d):
            if name == '':
                basename = 'weight'
                modname = 'weights'
            else:
                basename = f'{name}.weight'
                modname = f'{name}.weights'
            cnv_weights = state_dict[basename]
            del state_dict[basename]
            for j in range(switch_breadth):
                state_dict[f'{modname}.{j}'] = cnv_weights.clone()
    return state_dict


def test_net():
    base_conv = Conv2d(32, 64, 3, stride=2, padding=1, bias=True).to('cuda')
    mod_conv = SwitchedConv(32, 64, 3, switch_breadth=8, stride=2, padding=1, bias=True, include_coupler=True, coupler_dim_in=128).to('cuda')
    mod_sd = convert_conv_net_state_dict_to_switched_conv(base_conv, 8)
    mod_conv.load_state_dict(mod_sd, strict=False)
    inp = torch.randn((8,32,128,128), device='cuda')
    sel = torch.randn((8,128,32,32), device='cuda')
    out1 = base_conv(inp)
    out2 = mod_conv(inp, sel)
    assert(torch.max(torch.abs(out1-out2)) < 1e-6)

def perform_conversion():
    sd = torch.load("../experiments/rrdb_imgset_226500_generator.pth")
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in sd.items():
        if k.startswith('module.'):
            load_net_clean[k.replace('module.', '')] = v
        else:
            load_net_clean[k] = v
    sd = load_net_clean
    import models.RRDBNet_arch as rrdb
    block = functools.partial(rrdb.RRDBWithBypass)
    mod = rrdb.RRDBNet(in_channels=3, out_channels=3,
                                mid_channels=64, num_blocks=23, body_block=block, scale=2, initial_stride=2)
    mod.load_state_dict(sd)
    converted = convert_conv_net_state_dict_to_switched_conv(mod, 8, ['body.','conv_first','resnet_encoder'])
    torch.save(converted, "../experiments/rrdb_imgset_226500_generator_converted.pth")


if __name__ == '__main__':
    perform_conversion()
