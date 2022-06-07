from abc import abstractmethod

import math

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision  # For debugging, not actually used.
from x_transformers.x_transformers import RelativePositionBias

from models.audio.music.gpt_music import GptMusicLower
from models.audio.music.music_quantizer import MusicQuantizer
from models.diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
from models.diffusion.nn import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from models.lucidrains.x_transformers import Encoder
from trainer.networks import register_model
from utils.util import checkpoint, print_network, ceil_multiple


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

    def __init__(self, channels, use_conv, dims=2, out_channels=None, factor=None, ksize=3, pad=1):
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

    def __init__(self, channels, use_conv, dims=2, out_channels=None, factor=None, ksize=None, pad=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        if ksize is None:
            ksize = 3
            pad = 1
            if dims == 1:
                ksize = 5
                pad = 2

        if dims == 1:
            stride = 4
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


class ResBlock(TimestepBlock):
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
        emb_channels,
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
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = 1 if kernel_size == 3 else (2 if kernel_size == 5 else 0)

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

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
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
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, x, emb
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
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
        relative_pos_embeddings=False,
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
        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64)
        else:
            self.relative_pos_embeddings = None

    def forward(self, x, mask=None):
        if self.do_checkpoint:
            return checkpoint(self._forward, x, mask)
        else:
            return self._forward(x, mask)

    def _forward(self, x, mask):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, mask, self.relative_pos_embeddings)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, rel_pos=None):
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
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        if rel_pos is not None:
            weight = rel_pos(weight.reshape(bs, self.n_heads, weight.shape[-2], weight.shape[-1])).reshape(bs * self.n_heads, weight.shape[-2], weight.shape[-1])
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        if mask is not None:
            # The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs.
            mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
            weight = weight * mask
        a = th.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, rel_pos=None):
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
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        if rel_pos is not None:
            weight = rel_pos(weight)
        if mask is not None:
            # The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs.
            mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
            weight = weight * mask
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetMusicModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        input_vec_dim,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        ar_prior=False,
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_raw_y_as_embedding=False,
        unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.unconditioned_percentage = unconditioned_percentage
        self.ar_prior = ar_prior

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.ar_prior:
            self.ar_input = nn.Linear(input_vec_dim, model_channels)
            self.ar_prior_intg = Encoder(
                    dim=model_channels,
                    depth=4,
                    heads=num_heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                    zero_init_branch_output=True,
                    ff_mult=1,
                )
        else:
            self.input_converter = nn.Linear(input_vec_dim, model_channels)
            self.code_converter = Encoder(
                        dim=model_channels,
                        depth=4,
                        heads=num_heads,
                        ff_dropout=dropout,
                        attn_dropout=dropout,
                        use_rmsnorm=True,
                        ff_glu=True,
                        rotary_pos_emb=True,
                        zero_init_branch_output=True,
                        ff_mult=1,
                    )
        self.unconditioned_embedding = nn.Parameter(torch.randn(1,1,model_channels))
        self.x_processor = conv_nd(dims, in_channels, model_channels, 3, padding=1)

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        self.use_raw_y_as_embedding = use_raw_y_as_embedding
        assert not ((self.num_classes is not None) and use_raw_y_as_embedding)  # These are mutually-exclusive.

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, model_channels*2, model_channels, 1, padding=0)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, y, conditioning_free=False):
        orig_x_shape = x.shape[-1]
        cm = ceil_multiple(x.shape[-1], 16)
        if cm != 0:
            pc = (cm - x.shape[-1]) / x.shape[-1]
            x = F.pad(x, (0, cm - x.shape[-1]))
            y = F.pad(y.permute(0,2,1), (0, int(pc * y.shape[-1]))).permute(0,2,1)

        unused_params = []
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if conditioning_free:
            expanded_code_emb = self.unconditioned_embedding.repeat(x.shape[0], x.shape[-1], 1).permute(0,2,1)
            if self.ar_prior:
                unused_params.extend(list(self.ar_input.parameters()) + list(self.ar_prior_intg.parameters()))
            else:
                unused_params.extend(list(self.input_converter.parameters()) + list(self.code_converter.parameters()))
        else:
            code_emb = self.ar_input(y) if self.ar_prior else self.input_converter(y)
            if self.training and self.unconditioned_percentage > 0:
                unconditioned_batches = torch.rand((code_emb.shape[0], 1, 1),
                                                   device=code_emb.device) < self.unconditioned_percentage
                code_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(y.shape[0], 1, 1),
                                       code_emb)
            code_emb = self.ar_prior_intg(code_emb) if self.ar_prior else self.code_converter(code_emb)
            expanded_code_emb = F.interpolate(code_emb.permute(0,2,1), size=x.shape[-1], mode='nearest')

        h = x.type(self.dtype)
        expanded_code_emb = expanded_code_emb.type(self.dtype)

        h = self.x_processor(h)
        h = torch.cat([h, expanded_code_emb], dim=1)

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        h = h + extraneous_addition * 0

        out = self.out(h)
        return out[:, :, :orig_x_shape]


class UNetMusicModelWithQuantizer(nn.Module):
    def __init__(self, freeze_quantizer_until=20000, **kwargs):
        super().__init__()

        self.internal_step = 0
        self.freeze_quantizer_until = freeze_quantizer_until
        self.diff = UNetMusicModel(**kwargs)
        self.m2v = MusicQuantizer(inp_channels=256, inner_dim=[1024,1024,512], codevector_dim=1024, codebook_size=512, codebook_groups=2)
        self.m2v.quantizer.temperature = self.m2v.min_gumbel_temperature
        del self.m2v.up

    def update_for_step(self, step, *args):
        self.internal_step = step
        qstep = max(0, self.internal_step - self.freeze_quantizer_until)
        self.m2v.quantizer.temperature = max(
                    self.m2v.max_gumbel_temperature * self.m2v.gumbel_temperature_decay**qstep,
                    self.m2v.min_gumbel_temperature,
                )

    def forward(self, x, timesteps, truth_mel, disable_diversity=False, conditioning_input=None, conditioning_free=False):
        quant_grad_enabled = self.internal_step > self.freeze_quantizer_until
        with torch.set_grad_enabled(quant_grad_enabled):
            proj, diversity_loss = self.m2v(truth_mel, return_decoder_latent=True)
            proj = proj.permute(0,2,1)

        # Make sure this does not cause issues in DDP by explicitly using the parameters for nothing.
        if not quant_grad_enabled:
            unused = 0
            for p in self.m2v.parameters():
                unused = unused + p.mean() * 0
            proj = proj + unused
            diversity_loss = diversity_loss * 0

        diff = self.diff(x, timesteps, proj, conditioning_free=conditioning_free)
        if disable_diversity:
            return diff
        return diff, diversity_loss

    def get_debug_values(self, step, __):
        if self.m2v.total_codes > 0:
            return {'histogram_codes': self.m2v.codes[:self.m2v.total_codes]}
        else:
            return {}


class UNetMusicModelARPrior(nn.Module):
    def __init__(self, freeze_unet=False, **kwargs):
        super().__init__()

        self.internal_step = 0
        self.ar = GptMusicLower(dim=512, layers=12)
        for p in self.ar.parameters():
            p.DO_NOT_TRAIN = True
            p.requires_grad = False

        self.diff = UNetMusicModel(ar_prior=True, **kwargs)
        if freeze_unet:
            for p in self.diff.parameters():
                p.DO_NOT_TRAIN = True
                p.requires_grad = False
            for p in list(self.diff.ar_prior_intg.parameters()) + list(self.diff.ar_input.parameters()):
                del p.DO_NOT_TRAIN
                p.requires_grad = True

    def forward(self, x, timesteps, truth_mel, disable_diversity=False, conditioning_input=None, conditioning_free=False):
        with torch.no_grad():
            prior = self.ar(truth_mel, conditioning_input, return_latent=True)

        diff = self.diff(x, timesteps, prior, conditioning_free=conditioning_free)
        return diff


@register_model
def register_unet_diffusion_music_codes(opt_net, opt):
    return UNetMusicModelWithQuantizer(**opt_net['args'])

@register_model
def register_unet_diffusion_music_ar_prior(opt_net, opt):
    return UNetMusicModelARPrior(**opt_net['args'])


if __name__ == '__main__':
    clip = torch.randn(2, 256, 300)
    cond = torch.randn(2, 256, 300)
    ts = torch.LongTensor([600, 600])
    model = UNetMusicModelARPrior(in_channels=256, out_channels=512, model_channels=640, num_res_blocks=3, input_vec_dim=512,
                                  attention_resolutions=(2,4), channel_mult=(1,2,3), dims=1,
                                  use_scale_shift_norm=True, dropout=.1, num_heads=8, unconditioned_percentage=.4, freeze_unet=True)
    print_network(model)

    ar_weights = torch.load('D:\\dlas\\experiments\\train_music_gpt\\models\\44500_generator_ema.pth')
    model.ar.load_state_dict(ar_weights, strict=True)
    diff_weights = torch.load('X:\\dlas\\experiments\\train_music_diffusion_unet_music\\models\\55500_generator_ema.pth')
    pruned_diff_weights = {}
    for k,v in diff_weights.items():
        if k.startswith('diff.'):
            pruned_diff_weights[k.replace('diff.', '')] = v
    model.diff.load_state_dict(pruned_diff_weights, strict=False)
    torch.save(model.state_dict(), 'sample.pth')

    model(clip, ts, cond, cond)

