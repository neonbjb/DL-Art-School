import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from math import sqrt


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))


def laplace_smoothing(x, n_categories, eps = 1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(torch.pow(x, 2), dim=1, keepdims=True) + epsilon)


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
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


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


def default_init_weights(module, scale=1):
    """Initialize network weights.
    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :x.shape[-1]].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, factor=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if factor is None:
            if dims == 1:
                self.factor = 4
            else:
                self.factor = 2
        else:
            self.factor = factor
        if use_conv:
            ksize = 3
            pad = 1
            if dims == 1:
                ksize = 5
                pad = 2
            self.conv = conv_nd(dims, self.channels, self.out_channels, ksize, padding=pad)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        x = F.interpolate(x, scale_factor=self.factor, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, factor=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        ksize = 3
        pad = 1
        if dims == 1:
            stride = 4
            ksize = 5
            pad = 2
        elif dims == 2:
            stride = 2
        else:
            stride = (1,2,2)
        if factor is not None:
            stride = factor
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, ksize, stride=stride, padding=pad
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        kernel_size=3,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = 1 if kernel_size == 3 else 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, x, emb
        )

    def _forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
        do_checkpoint=True,
    ):
        super().__init__()
        self.channels = channels
        self.do_checkpoint = do_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, mask=None):
        if self.do_checkpoint:
            return checkpoint(self._forward, x, mask)
        else:
            return self._forward(x, mask)

    def _forward(self, x, mask):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, mask)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        if mask is not None:
            # The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs.
            mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
            weight = weight * mask
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        if mask is not None:
            # The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs.
            mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
            weight = weight * mask
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


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


# simply define a silu function
def silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input)

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class SiLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return silu(input)


''' Convenience class with Conv->BN->ReLU. Includes weight initialization and auto-padding for standard
    kernel sizes. '''
class ConvBnRelu(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3, stride=1, activation=True, norm=True, bias=True):
        super(ConvBnRelu, self).__init__()
        padding_map = {1: 0, 3: 1, 5: 2, 7: 3}
        assert kernel_size in padding_map.keys()
        self.conv = nn.Conv2d(filters_in, filters_out, kernel_size, stride, padding_map[kernel_size], bias=bias)
        if norm:
            self.bn = nn.BatchNorm2d(filters_out)
        else:
            self.bn = None
        if activation:
            self.relu = nn.ReLU()
        else:
            self.relu = None

        # Init params.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu' if self.relu else 'linear')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            return self.relu(x)
        else:
            return x


''' Convenience class with Conv->BN->SiLU. Includes weight initialization and auto-padding for standard
    kernel sizes. '''
class ConvBnSilu(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3, stride=1, activation=True, norm=True, bias=True, weight_init_factor=1):
        super(ConvBnSilu, self).__init__()
        padding_map = {1: 0, 3: 1, 5: 2, 7: 3}
        assert kernel_size in padding_map.keys()
        self.conv = nn.Conv2d(filters_in, filters_out, kernel_size, stride, padding_map[kernel_size], bias=bias)
        if norm:
            self.bn = nn.BatchNorm2d(filters_out)
        else:
            self.bn = None
        if activation:
            self.silu = SiLU()
        else:
            self.silu = None

        # Init params.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu' if self.silu else 'linear')
                m.weight.data *= weight_init_factor
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.silu:
            return self.silu(x)
        else:
            return x


''' Convenience class with Conv->BN->LeakyReLU. Includes weight initialization and auto-padding for standard
    kernel sizes. '''
class ConvBnLelu(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3, stride=1, activation=True, norm=True, bias=True, weight_init_factor=1):
        super(ConvBnLelu, self).__init__()
        padding_map = {1: 0, 3: 1, 5: 2, 7: 3}
        assert kernel_size in padding_map.keys()
        self.conv = nn.Conv2d(filters_in, filters_out, kernel_size, stride, padding_map[kernel_size], bias=bias)
        if norm:
            self.bn = nn.BatchNorm2d(filters_out)
        else:
            self.bn = None
        if activation:
            self.lelu = nn.LeakyReLU(negative_slope=.1)
        else:
            self.lelu = None

        # Init params.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
        if self.bn:
            x = self.bn(x)
        if self.lelu:
            return self.lelu(x)
        else:
            return x


''' Convenience class with Conv->GroupNorm->LeakyReLU. Includes weight initialization and auto-padding for standard
    kernel sizes. '''
class ConvGnLelu(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3, stride=1, activation=True, norm=True, bias=True, num_groups=8, weight_init_factor=1):
        super(ConvGnLelu, self).__init__()
        padding_map = {1: 0, 3: 1, 5: 2, 7: 3}
        assert kernel_size in padding_map.keys()
        self.conv = nn.Conv2d(filters_in, filters_out, kernel_size, stride, padding_map[kernel_size], bias=bias)
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
            if isinstance(m, nn.Conv2d):
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


''' Convenience class with Conv->BN->SiLU. Includes weight initialization and auto-padding for standard
    kernel sizes. '''
class ConvGnSilu(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3, stride=1, activation=True, norm=True, bias=True, num_groups=8, weight_init_factor=1, convnd=nn.Conv2d):
        super(ConvGnSilu, self).__init__()
        padding_map = {1: 0, 3: 1, 5: 2, 7: 3}
        assert kernel_size in padding_map.keys()
        self.conv = convnd(filters_in, filters_out, kernel_size, stride, padding_map[kernel_size], bias=bias)
        if norm:
            self.gn = nn.GroupNorm(num_groups, filters_out)
        else:
            self.gn = None
        if activation:
            self.silu = SiLU()
        else:
            self.silu = None

        # Init params.
        for m in self.modules():
            if isinstance(m, convnd):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu' if self.silu else 'linear')
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
        if self.silu:
            return self.silu(x)
        else:
            return x


''' Convenience class with Conv->BN->ReLU. Includes weight initialization and auto-padding for standard
    kernel sizes. '''
class ConvBnRelu(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3, stride=1, activation=True, norm=True, bias=True, weight_init_factor=1):
        super(ConvBnRelu, self).__init__()
        padding_map = {1: 0, 3: 1, 5: 2, 7: 3}
        assert kernel_size in padding_map.keys()
        self.conv = nn.Conv2d(filters_in, filters_out, kernel_size, stride, padding_map[kernel_size], bias=bias)
        if norm:
            self.bn = nn.BatchNorm2d(filters_out)
        else:
            self.bn = None
        if activation:
            self.relu = nn.ReLU()
        else:
            self.relu = None

        # Init params.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu' if self.relu else 'linear')
                m.weight.data *= weight_init_factor
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            return self.relu(x)
        else:
            return x


# Simple way to chain multiple conv->act->norms together in an intuitive way.
class MultiConvBlock(nn.Module):
    def __init__(self, filters_in, filters_mid, filters_out, kernel_size, depth, scale_init=1, norm=False, weight_init_factor=1):
        assert depth >= 2
        super(MultiConvBlock, self).__init__()
        self.noise_scale = nn.Parameter(torch.full((1,), fill_value=.01))
        self.bnconvs = nn.ModuleList([ConvBnLelu(filters_in, filters_mid, kernel_size, norm=norm, bias=False, weight_init_factor=weight_init_factor)] +
                                     [ConvBnLelu(filters_mid, filters_mid, kernel_size, norm=norm, bias=False, weight_init_factor=weight_init_factor) for i in range(depth - 2)] +
                                     [ConvBnLelu(filters_mid, filters_out, kernel_size, activation=False, norm=False, bias=False, weight_init_factor=weight_init_factor)])
        self.scale = nn.Parameter(torch.full((1,), fill_value=scale_init, dtype=torch.float))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is not None:
            noise = noise * self.noise_scale
            x = x + noise
        for m in self.bnconvs:
            x = m.forward(x)
        return x * self.scale + self.bias


# Block that upsamples 2x and reduces incoming filters by 2x. It preserves structure by taking a passthrough feed
# along with the feature representation.
class ExpansionBlock(nn.Module):
    def __init__(self, filters_in, filters_out=None, block=ConvGnSilu):
        super(ExpansionBlock, self).__init__()
        if filters_out is None:
            filters_out = filters_in // 2
        self.decimate = block(filters_in, filters_out, kernel_size=1, bias=False, activation=False, norm=True)
        self.process_passthrough = block(filters_out, filters_out, kernel_size=3, bias=True, activation=False, norm=True)
        self.conjoin = block(filters_out*2, filters_out, kernel_size=3, bias=False, activation=True, norm=False)
        self.process = block(filters_out, filters_out, kernel_size=3, bias=False, activation=True, norm=True)

    # input is the feature signal with shape  (b, f, w, h)
    # passthrough is the structure signal with shape (b, f/2, w*2, h*2)
    # output is conjoined upsample with shape (b, f/2, w*2, h*2)
    def forward(self, input, passthrough):
        x = F.interpolate(input, scale_factor=2, mode="nearest")
        x = self.decimate(x)
        p = self.process_passthrough(passthrough)
        x = self.conjoin(torch.cat([x, p], dim=1))
        return self.process(x)


# Block that upsamples 2x and reduces incoming filters by 2x. It preserves structure by taking a passthrough feed
# along with the feature representation.
# Differs from ExpansionBlock because it performs all processing in 2xfilter space and decimates at the last step.
class ExpansionBlock2(nn.Module):
    def __init__(self, filters_in, filters_out=None, block=ConvGnSilu):
        super(ExpansionBlock2, self).__init__()
        if filters_out is None:
            filters_out = filters_in // 2
        self.decimate = block(filters_in, filters_out, kernel_size=1, bias=False, activation=False, norm=True)
        self.process_passthrough = block(filters_out, filters_out, kernel_size=3, bias=True, activation=False, norm=True)
        self.conjoin = block(filters_out*2, filters_out*2, kernel_size=3, bias=False, activation=True, norm=False)
        self.reduce = block(filters_out*2, filters_out, kernel_size=3, bias=False, activation=True, norm=True)

    # input is the feature signal with shape  (b, f, w, h)
    # passthrough is the structure signal with shape (b, f/2, w*2, h*2)
    # output is conjoined upsample with shape (b, f/2, w*2, h*2)
    def forward(self, input, passthrough):
        x = F.interpolate(input, scale_factor=2, mode="nearest")
        x = self.decimate(x)
        p = self.process_passthrough(passthrough)
        x = self.conjoin(torch.cat([x, p], dim=1))
        return self.reduce(x)


# Similar to ExpansionBlock2 but does not upsample.
class ConjoinBlock(nn.Module):
    def __init__(self, filters_in, filters_out=None, filters_pt=None, block=ConvGnSilu, norm=True):
        super(ConjoinBlock, self).__init__()
        if filters_out is None:
            filters_out = filters_in
        if filters_pt is None:
            filters_pt = filters_in
        self.process = block(filters_in + filters_pt, filters_in + filters_pt, kernel_size=3, bias=False, activation=True, norm=norm)
        self.decimate = block(filters_in + filters_pt, filters_out, kernel_size=1, bias=False, activation=False, norm=norm)

    def forward(self, input, passthrough):
        x = torch.cat([input, passthrough], dim=1)
        x = self.process(x)
        return self.decimate(x)


# Designed explicitly to join a mainline trunk with reference data. Implemented as a residual branch.
class ReferenceJoinBlock(nn.Module):
    def __init__(self, nf, residual_weight_init_factor=1, block=ConvGnLelu, final_norm=False, kernel_size=3, depth=3, join=True):
        super(ReferenceJoinBlock, self).__init__()
        self.branch = MultiConvBlock(nf * 2, nf + nf // 2, nf, kernel_size=kernel_size, depth=depth,
                                     scale_init=residual_weight_init_factor, norm=False,
                                     weight_init_factor=residual_weight_init_factor)
        if join:
            self.join_conv = block(nf, nf, kernel_size=kernel_size, norm=final_norm, bias=False, activation=True)
        else:
            self.join_conv = None

    def forward(self, x, ref):
        joined = torch.cat([x, ref], dim=1)
        branch = self.branch(joined)
        if self.join_conv is not None:
            return self.join_conv(x + branch), torch.std(branch)
        else:
            return x + branch, torch.std(branch)


# Basic convolutional upsampling block that uses interpolate.
class UpconvBlock(nn.Module):
    def __init__(self, filters_in, filters_out=None, block=ConvGnSilu, norm=True, activation=True, bias=False):
        super(UpconvBlock, self).__init__()
        self.process = block(filters_in, filters_out, kernel_size=3, bias=bias, activation=activation, norm=norm)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.process(x)


# Scales an image up 2x and performs intermediary processing. Designed to be the final block in an SR network.
class FinalUpsampleBlock2x(nn.Module):
    def __init__(self, nf, block=ConvGnLelu, out_nc=3, scale=2):
        super(FinalUpsampleBlock2x, self).__init__()
        if scale == 2:
            self.chain = nn.Sequential(block(nf, nf, kernel_size=3, norm=False, activation=True, bias=True),
                                       UpconvBlock(nf, nf // 2, block=block, norm=False, activation=True, bias=True),
                                       block(nf // 2, nf // 2, kernel_size=3, norm=False, activation=False, bias=True),
                                       block(nf // 2, out_nc, kernel_size=3, norm=False, activation=False, bias=False))
        else:
            self.chain = nn.Sequential(block(nf, nf, kernel_size=3, norm=False, activation=True, bias=True),
                                       UpconvBlock(nf, nf, block=block, norm=False, activation=True, bias=True),
                                       block(nf, nf, kernel_size=3, norm=False, activation=False, bias=True),
                                       UpconvBlock(nf, nf // 2, block=block, norm=False, activation=True, bias=True),
                                       block(nf // 2, nf // 2, kernel_size=3, norm=False, activation=False, bias=True),
                                       block(nf // 2, out_nc, kernel_size=3, norm=False, activation=False, bias=False))

    def forward(self, x):
        return self.chain(x)
