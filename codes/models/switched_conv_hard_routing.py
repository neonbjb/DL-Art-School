import math

import torch
import torch.nn as nn
import switched_conv_cuda_naive
from lambda_networks import LambdaLayer
from torch.nn import init, Conv2d, MSELoss
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist

from trainer.losses import ConfigurableLoss


def SwitchedConvRoutingNormal(input, selector, weight, bias, stride=1):
    convs = []
    b, s, h, w = selector.shape
    for sel in range(s):
        convs.append(F.conv2d(input, weight[:, :, sel, :, :], bias, stride=stride, padding=weight.shape[-1] // 2))
    output = torch.stack(convs, dim=1) * selector.unsqueeze(dim=2)
    return output.sum(dim=1)


class SwitchedConvHardRoutingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, selector, weight, bias, stride=1):
        # Build hard attention mask from selector input
        b, s, h, w = selector.shape

        mask = selector.argmax(dim=1).int()
        output = switched_conv_cuda_naive.forward(input, mask, weight, bias, stride)

        ctx.stride = stride
        ctx.breadth = s
        ctx.save_for_backward(*[input, mask, weight, bias])
        return output

    @staticmethod
    def backward(ctx, grad):
        input, mask, weight, bias = ctx.saved_tensors
        grad, grad_sel, grad_w, grad_b = switched_conv_cuda_naive.backward(input, grad.contiguous(), mask, weight, bias, ctx.stride)
        return grad, grad_sel, grad_w, grad_b, None


# Implements KeepTopK where k=1 from mixture of experts paper.
class KeepTop1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        mask = torch.nn.functional.one_hot(input.argmax(dim=1), num_classes=input.shape[1]).permute(0,3,1,2)
        input[mask != 1] = -float('inf')
        ctx.save_for_backward(mask)
        return input

    @staticmethod
    def backward(ctx, grad):
        import pydevd
        pydevd.settrace(suspend=False, trace_only_current_thread=True)
        mask = ctx.saved_tensors
        grad_input = grad.clone()
        grad_input[mask != 1] = 0
        return grad_input


class RouteTop1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        mask = torch.nn.functional.one_hot(input.argmax(dim=1), num_classes=input.shape[1]).permute(0,3,1,2)
        out = torch.ones_like(input)
        out[mask != 1] = 0
        ctx.save_for_backward(mask, input.clone())
        return out

    @staticmethod
    def backward(ctx, grad):
        # Enable breakpoints in this function:  (Comment out if not debugging)
        #import pydevd
        #pydevd.settrace(suspend=False, trace_only_current_thread=True)

        mask, input = ctx.saved_tensors
        input[mask != 1] = 1
        grad_input = grad.clone()
        grad_input[mask != 1] = 0
        grad_input_n = grad_input / input  # Above, we made everything either a zero or a one. Unscale the ones by dividing by the unmasked inputs.
        return grad_input_n


"""
SwitchNorm is meant to be applied against the Softmax output of an switching function across a large set of
switch computations. It is meant to promote an equal distribution of switch weights by decreasing the magnitude
of switch weights that are over-used and increasing the magnitude of under-used weights.

The return value has the exact same format as a normal Softmax output and can be used directly into the input of an
switch equation.

Since the whole point of convolutional switch is to enable training extra-wide networks to operate on a large number
of image categories, it makes almost no sense to perform this type of norm against a single mini-batch of images: some
of the switches will not be used in such a small context - and that's good! This is solved by accumulating. Every 
forward pass computes a norm across the current minibatch. That norm is added into a rotating buffer of size 
<accumulator_size>. The actual normalization occurs across the entire rotating buffer.

You should set accumulator size according to two factors:
- Your batch size. Smaller batch size should mean greater accumulator size.
- Your image diversity. More diverse images have less need for the accumulator.
- How wide your switch/switching group size is. More groups mean you're going to want more accumulation.

Note: This norm makes the (potentially flawed) assumption that each forward() pass has unique data. For maximum 
      effectiveness, avoid doing this - or make alterations to work around it.
Note: This norm does nothing for the first <accumulator_size> iterations.
"""
class SwitchNorm(nn.Module):
    def __init__(self, group_size, accumulator_size=128):
        super().__init__()
        self.accumulator_desired_size = accumulator_size
        self.group_size = group_size
        self.register_buffer("accumulator_index", torch.zeros(1, dtype=torch.long, device='cpu'))
        self.register_buffer("accumulator_filled", torch.zeros(1, dtype=torch.long, device='cpu'))
        self.register_buffer("accumulator", torch.zeros(accumulator_size, group_size))

    def add_norm_to_buffer(self, x):
        flat = x.sum(dim=[0, 2, 3])
        norm = flat / torch.mean(flat)

        self.accumulator[self.accumulator_index] = norm.detach().clone()
        self.accumulator_index += 1
        if self.accumulator_index >= self.accumulator_desired_size:
            self.accumulator_index *= 0
            if self.accumulator_filled <= 0:
                self.accumulator_filled += 1

    # Input into forward is a switching tensor of shape (batch,groups,width,height)
    def forward(self, x: torch.Tensor, update_attention_norm=True):
        assert len(x.shape) == 4

        # Push the accumulator to the right device on the first iteration.
        if self.accumulator.device != x.device:
            self.accumulator = self.accumulator.to(x.device)

        # In eval, don't change the norm buffer.
        if self.training and update_attention_norm:
            self.add_norm_to_buffer(x)

        # Reduce across all distributed entities, if needed
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.accumulator, op=dist.ReduceOp.SUM)
            self.accumulator /= dist.get_world_size()

        # Compute the norm factor.
        if self.accumulator_filled > 0:
            norm = torch.mean(self.accumulator, dim=0)
        else:
            norm = torch.ones(self.group_size, device=self.accumulator.device)
        x = x / norm.view(1,-1,1,1)

        # Need to re-normalize x so that the groups dimension sum to 1, just like when it was fed in.
        return x / x.sum(dim=1, keepdim=True)


class MixtureOfExperts2dRouter(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.wnoise = nn.Parameter(torch.zeros(1,num_experts,1,1))
        self.wg = nn.Parameter(torch.zeros(1,num_experts,1,1))

    def forward(self, x):
        wg = x * self.wg
        wnoise = nn.functional.softplus(x * self.wnoise)
        H = wg + torch.randn_like(x) * wnoise

        # Produce the load-balancing loss.
        eye = torch.eye(self.num_experts, device=x.device).view(1,self.num_experts,self.num_experts,1,1)
        mask=torch.abs(1-eye)
        b,c,h,w=H.shape
        ninf = torch.zeros_like(eye)
        ninf[eye==1] = -float('inf')
        H_masked=H.view(b,c,1,h,w)*mask+ninf   # ninf is necessary because otherwise torch.max() will not pick up negative numbered maxes.
        max_excluding=torch.max(H_masked,dim=2)[0]

        # load_loss and G are stored as local members to facilitate their use by hard routing regularization losses.
        # this is a risky op - it can easily result in memory leakage. Clients *must* use self.reset() below.
        self.load_loss = torch.erf((wg - max_excluding)/wnoise)
        #self.G = nn.functional.softmax(KeepTop1.apply(H), dim=1)  The paper proposes this equation, but performing a softmax on a Top-1 per the paper results in zero gradients into H, so:
        self.G = RouteTop1.apply(nn.functional.softmax(H, dim=1))  # This variant can route gradients downstream.

        return self.G

    # Retrieve the locally stored loss values and delete them from membership (so as to not waste memory)
    def reset(self):
        G, load = self.G, self.load_loss
        del self.G
        del self.load_loss
        return G, load


# Loss that finds instances of MixtureOfExperts2dRouter in the given network and extracts their custom losses.
class MixtureOfExpertsLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.routers = []  # This is filled in during the first forward() pass and cached from there.
        self.first_forward_encountered = False
        self.load_weight = opt['load_weight']
        self.importance_weight = opt['importance_weight']

    def forward(self, net, state):
        if not self.first_forward_encountered:
            for m in net.modules():
                if isinstance(m, MixtureOfExperts2dRouter):
                    self.routers.append(m)
            self.first_forward_encountered = True

        l_importance = 0
        l_load = 0
        for r in self.routers:
            G, L = r.reset()
            l_importance += G.var().square()
            l_load += L.var().square()
        return l_importance * self.importance_weight + l_load * self.load_weight


class SwitchTransformersLoadBalancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = SwitchNorm(8, accumulator_size=256)

    def forward(self, x):
        self.soft = self.norm(nn.functional.softmax(x, dim=1))
        self.hard = RouteTop1.apply(self.soft)  # This variant can route gradients downstream.
        return self.hard

    def reset(self):
        soft, hard = self.soft, self.hard
        del self.soft, self.hard
        return soft, hard


class SwitchTransformersLoadBalancingLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.routers = []  # This is filled in during the first forward() pass and cached from there.
        self.first_forward_encountered = False

    def forward(self, net, state):
        if not self.first_forward_encountered:
            for m in net.modules():
                if isinstance(m, SwitchTransformersLoadBalancer):
                    self.routers.append(m)
            self.first_forward_encountered = True

        loss = 0
        for r in self.routers:
            soft, hard = r.reset()
            N = hard.shape[1]
            h_mean = hard.mean(dim=[0,2,3])
            s_mean = soft.mean(dim=[0,2,3])
            loss += torch.dot(h_mean, s_mean) * N
        return loss


class SwitchedConvHardRouting(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 kernel_sz,
                 breadth,
                 stride=1,
                 bias=True,
                 dropout_rate=0.0,
                 include_coupler: bool = False,  # A 'coupler' is a latent converter which can make any bxcxhxw tensor a compatible switchedconv selector by performing a linear 1x1 conv, softmax and interpolate.
                 coupler_mode: str = 'standard',
                 coupler_dim_in: int = 0):
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
                self.coupler = Conv2d(coupler_dim_in, breadth, kernel_size=1, stride=self.stride)
            elif coupler_mode == 'lambda':
                self.coupler = nn.Sequential(nn.Conv2d(coupler_dim_in, coupler_dim_in, 1),
                                             nn.BatchNorm2d(coupler_dim_in),
                                             nn.ReLU(),
                                             LambdaLayer(dim=coupler_dim_in, dim_out=breadth, r=23, dim_k=16, heads=2, dim_u=1),
                                             nn.BatchNorm2d(breadth),
                                             nn.ReLU(),
                                             Conv2d(breadth, breadth, 1, stride=self.stride))
        else:
            self.coupler = None
        #self.gate = MixtureOfExperts2dRouter(breadth)
        self.gate = SwitchTransformersLoadBalancer()

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
            selector = self.coupler(selector)
        assert selector is not None

        # Apply dropout at the batch level per kernel.
        if self.training and self.dropout_rate > 0:
            b, c, h, w = selector.shape
            drop = torch.rand((b, c, 1, 1), device=input.device) > self.dropout_rate
            # Ensure that there is always at least one switch left un-dropped out
            fix_blank = (drop.sum(dim=1, keepdim=True) == 0).repeat(1, c, 1, 1)
            drop = drop.logical_or(fix_blank)
            selector = drop * selector

        selector = self.gate(selector)

        # Debugging variables
        self.last_select = selector.detach().clone()
        self.latest_masks = (selector.max(dim=1, keepdim=True)[0].repeat(1,self.breadth,1,1) == selector).float().argmax(dim=1)

        if False:
            # This is a custom CUDA implementation which should be faster and less memory intensive (once completed).
            return SwitchedConvHardRoutingFunction.apply(input, selector, self.weight, self.bias, self.stride)
        else:
            # This composes the switching functionality using raw Torch, which basically consists of computing each of <breadth> convs separately and combining them.
            return SwitchedConvRoutingNormal(input, selector, self.weight, self.bias, self.stride)


# Given a state_dict and the module that that sd belongs to, strips out all Conv2d.weight parameters and replaces them
# with the equivalent SwitchedConv.weight parameters. Does not create coupler params.
def convert_conv_net_state_dict_to_switched_conv(module, switch_breadth, ignore_list=[]):
    state_dict = module.state_dict()
    for name, m in module.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        ignored = False
        for smod in ignore_list:
            if smod in name:
                ignored = True
                continue
        if ignored:
            continue
        if name == '':
            key = 'weight'
        else:
            key = f'{name}.weight'
        state_dict[key] = state_dict[key].unsqueeze(2).repeat(1,1,switch_breadth,1,1)
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