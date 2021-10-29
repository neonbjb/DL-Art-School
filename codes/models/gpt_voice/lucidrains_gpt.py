from inspect import isfunction

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

# helpers
from models.arch_util import checkpoint
from models.gpt_voice.reversible import ReversibleSequence, SequentialSequence
from utils.util import sequential_checkpoint


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, depth = 1):
    if isinstance(val, list):
        val = tuple(val)
    return val if isinstance(val, tuple) else (val,) * depth


class DivideMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim = self.dim, keepdim = True)
        return x / maxes


# https://arxiv.org/abs/2103.17239
class LayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout = 0., mult = 4.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, only_last_two_elements=False):
        if only_last_two_elements:
            h = x[:, -2:]
            h = self.net(h)
            return torch.cat([x[:, :-2], h], dim=1)
        else:
            return self.net(x)


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def stable_softmax(t, dim = -1, alpha = 32 ** 2):
    t = t / alpha
    t = t - torch.amax(t, dim = dim, keepdim = True)
    return (t * alpha).softmax(dim = dim)


# classes
class Attention(nn.Module):
    def __init__(self, dim, seq_len, non_causal_sequence_partition = 0, heads = 8, dim_head = 64, dropout = 0., stable = False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head ** -0.5

        self.stable = stable
        self.non_causal_sequence_partition = non_causal_sequence_partition

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None, only_last_two_elements=False):
        b, n, _, h, device = *x.shape, self.heads, x.device
        softmax = torch.softmax if not self.stable else stable_softmax

        # TODO: Q and V do not need to be recomputed for existing elements in intermediate_latents is specified. V would need to be cached though.
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = q * self.scale

        if only_last_two_elements:
            q = q[:, :, -2:]
            assert not exists(mask)  # Don't know how to resolve this (currently)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = max_neg_value(dots)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        i, j = dots.shape[-2:]
        mask = torch.ones(i, j, device = device).triu_(j - i + 1)
        if self.non_causal_sequence_partition > 0:
            non_causal_mask = torch.ones((i, j), device=device)
            non_causal_mask[:, :self.non_causal_sequence_partition] = 0
            mask = mask * non_causal_mask

        dots.masked_fill_(mask.bool(), mask_value)

        attn = softmax(dots, dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        seq_len,
        reversible = False,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        sparse_attn = False,
        stable = False,
        non_causal_sequence_partition=0,
    ):
        super().__init__()
        layers = nn.ModuleList([])
        sparse_layer = cast_tuple(sparse_attn, depth)

        for ind, sparse_attn in zip(range(depth), sparse_layer):
            attn = Attention(dim, stable=stable, non_causal_sequence_partition = non_causal_sequence_partition, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout)

            layers.append(nn.ModuleList([
                LayerScale(dim, ind + 1, PreNorm(dim, attn)),
                LayerScale(dim, ind + 1, PreNorm(dim, FeedForward(dim, mult = ff_mult, dropout = ff_dropout)))
            ]))

        # TODO: Remove this nonsense. I don't want to support reversible sequences and this is just a mess.
        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn}

        self.layers = execute_type(layers, args_route = attn_route_map, checkpoint=True)
        self.depth = depth

    def forward(self, x, return_intermediates=False):
        intermediates = []
        for attn, ff in self.layers.layers:
            x_ff = x + checkpoint(attn, x)
            x = x + ff(x_ff)
            if return_intermediates:
                intermediates.append((x_ff, x))
        if return_intermediates:
            return x, intermediates
        else:
            return x

    def infer_last_two(self, x, prev_intermediates):
        """
        Performs an forward pass only on the last two element in the given sequence (allowing it to attend to all other
        elements). This is useful for faster autoregressive decoding.

        The last two elements are important because in inference, the last element is the prediction candidate and the
        second-to-last element is a newly selected element from the autoregressive searching process.
        """
        assert(len(prev_intermediates) == self.depth)
        new_intermediates = []
        for (attn, ff), (int_ff, int_out) in zip(self.layers.layers, prev_intermediates):
            x = x + attn(x, only_last_two_elements=True)
            # Note that (x) is now only the last two element in the set. Conjoin it with the int_ff latent to compute the norm.
            x_ff = torch.cat([int_ff[:,:-1], x], dim=1)
            x = x + ff(x_ff, only_last_two_elements=True)
            new_intermediates.append((x_ff, x))
        return x, new_intermediates
