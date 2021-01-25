import math

import torch
import torch.nn as nn
import switched_conv_cuda_naive
from lambda_networks import LambdaLayer
from torch.nn import init, Conv2d, MSELoss
import torch.nn.functional as F
from tqdm import tqdm


class SwitchedConvHardRoutingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, selector, weight, bias, stride=1):
        # Build hard attention mask from selector input
        b, s, h, w = selector.shape
        selector_mask = (selector.max(dim=1, keepdim=True)[0].repeat(1,s,1,1) == selector).float()
        mask = selector_mask.argmax(dim=1).int()

        # Compute the convolution using the mask.
        outputs = switched_conv_cuda_naive.forward(input, mask, weight, bias, stride)
        ctx.stride = stride
        ctx.breadth = s
        ctx.save_for_backward(*[input, mask, weight, bias])
        return outputs

    @staticmethod
    def backward(ctx, grad):
        input, mask, weight, bias = ctx.saved_tensors

        # Get the grads for the convolution.
        grad, grad_w, grad_b = switched_conv_cuda_naive.backward(input, grad.contiguous(), mask, weight, bias, ctx.stride)

        # Get the selector grads
        selector_mask = torch.eye(ctx.breadth, device=input.device)[mask.long()].permute(0,3,1,2).unsqueeze(2)  # Note that this is not necessarily equivalent to the selector_mask from above, because under certain circumstances, two values could take on the value '1' in the above instance, whereas this is a true one-hot representation.
        grad_sel = ((grad * input).unsqueeze(1) * selector_mask).sum(2)
        return grad, grad_sel, grad_w, grad_b, None


class SwitchedConvHardRouting(nn.Module):
    def __init__(self, in_c, out_c, kernel_sz, breadth, stride=1, bias=True, dropout_rate=0.0,
        include_coupler: bool = False,  # A 'coupler' is a latent converter which can make any bxcxhxw tensor a compatible switchedconv selector by performing a linear 1x1 conv, softmax and interpolate.
        coupler_mode: str = 'standard',
        coupler_dim_in: int = 0,):
        super().__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.kernel_size = kernel_sz
        self.stride = stride
        self.has_bias = bias
        self.breadth = breadth
        self.dropout_rate = dropout_rate

        if include_coupler:
            if coupler_mode == 'standard':
                self.coupler = Conv2d(coupler_dim_in, breadth, kernel_size=1)
            elif coupler_mode == 'lambda':
                self.coupler = LambdaLayer(dim=coupler_dim_in, dim_out=breadth, r=23, dim_k=16, heads=2, dim_u=1)
        else:
            self.coupler = None

        self.weight = nn.Parameter(torch.empty(out_c, in_c, breadth, kernel_sz, kernel_sz))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_c))
        else:
            self.bias = torch.zeros(out_c)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[:,:,0,:,:])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def load_weights_from_conv(self, cnv):
        sd = cnv.state_dict()
        sd['weight'] = sd['weight'].unsqueeze(2).repeat(1,1,self.breadth,1,1)
        self.load_state_dict(sd)

    def forward(self, input, selector=None):
        if self.bias.device != input.device:
            self.bias = self.bias.to(input.device)  # Because this bias can be a tensor that is not moved with the rest of the module.

        # If a coupler was specified, run that to convert selector into a softmax distribution.
        if self.coupler:
            if selector is None:  # A coupler can convert from any input to a selector, so 'None' is allowed.
                selector = input
            selector = F.softmax(self.coupler(selector), dim=1)
            self.last_select = selector.detach().clone()
        assert selector is not None

        # Apply dropout at the batch level per kernel.
        if self.training and self.dropout_rate > 0:
            b, c, h, w = selector.shape
            drop = torch.rand((b, c, 1, 1), device=input.device) > self.dropout_rate
            # Ensure that there is always at least one switch left un-dropped out
            fix_blank = (drop.sum(dim=1, keepdim=True) == 0).repeat(1, c, 1, 1)
            drop = drop.logical_or(fix_blank)
            selector = drop * selector

        return SwitchedConvHardRoutingFunction.apply(input, selector, self.weight, self.bias, self.stride)


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
            state_dict[f'{name}.weight'] = state_dict[f'{name}.weight'].unsqueeze(2).repeat(1,1,switch_breadth,1,1)
    return state_dict


def test_net():
    for j in tqdm(range(100)):
        base_conv = Conv2d(32, 64, 3, stride=2, padding=1, bias=True).to('cuda')
        mod_conv = SwitchedConvHardRouting(32, 64, 3, breadth=8, stride=2, bias=True, include_coupler=True, coupler_dim_in=32, dropout_rate=.2).to('cuda')
        mod_sd = convert_conv_net_state_dict_to_switched_conv(base_conv, 8)
        mod_conv.load_state_dict(mod_sd, strict=False)
        inp = torch.randn((128,32,128,128), device='cuda')
        out1 = base_conv(inp)
        out2 = mod_conv(inp, None)
        compare = (out2+torch.rand_like(out2)*1e-6).detach()
        MSELoss()(out2, compare).backward()
        assert(torch.max(torch.abs(out1-out2)) < 1e-5)

if __name__ == '__main__':
    test_net()