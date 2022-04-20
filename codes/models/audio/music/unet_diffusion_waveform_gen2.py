import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from x_transformers import Encoder

from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import AttentionBlock, TimestepEmbedSequential, \
    Downsample, Upsample, TimestepBlock
from models.audio.tts.mini_encoder import AudioMiniEncoder
from models.audio.tts.unet_diffusion_tts7 import CheckpointedXTransformerEncoder
from scripts.audio.gen.use_diffuse_tts import ceil_multiple
from trainer.networks import register_model
from utils.util import checkpoint

def is_sequence(t):
    return t.dtype == torch.long


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        kernel_size=3,
        efficient_config=True,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = {1: 0, 3: 1, 5: 2}[kernel_size]
        eff_kernel = 1 if efficient_config else 3
        eff_padding = 0 if efficient_config else 1

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, eff_kernel, padding=eff_padding),
        )

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
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, eff_kernel, padding=eff_padding)

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
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class ResBlockSimple(nn.Module):
    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        dims=1,
        kernel_size=3,
        efficient_config=True,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        padding = {1: 0, 3: 1, 5: 2}[kernel_size]
        eff_kernel = 1 if efficient_config else 3
        eff_padding = 0 if efficient_config else 1

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, eff_kernel, padding=eff_padding),
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
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, eff_kernel, padding=eff_padding)

    def forward(self, x):
        return checkpoint(
            self._forward, x
        )

    def _forward(self, x):
        h = self.in_layers(x)
        return self.skip_connection(x) + h


class StructuralProcessor(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        #                256,128,64,32,16,8,4,2,1
        level_resblocks = [3,  3, 2, 2, 2,1,1,1]
        level_ch_div =    [1,  1, 2, 4, 4,8,8,16]
        levels = []
        lastdiv = 1
        for resblks, chdiv in zip(level_resblocks, level_ch_div):
            levels.append(nn.Sequential(*([nn.Conv1d(channels//lastdiv, channels//chdiv, 1)] +
                                          [ResBlockSimple(channels//chdiv, dropout) for _ in range(resblks)])))
            lastdiv = chdiv
        self.levels = nn.ModuleList(levels)

    def forward(self, x):
        h = x
        for level in self.levels:
            h = level(h)
            h = F.interpolate(h, scale_factor=2, mode='linear')
        return h


class DiffusionTts(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    Customized to be conditioned on an aligned prior derived from a autoregressive
    GPT-style model.

    :param in_channels: channels in the input Tensor.
    :param in_latent_channels: channels from the input latent.
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
            model_channels,
            in_channels=1,
            in_mel_channels=120,
            conditioning_dim_factor=8,
            out_channels=2,  # mean and variance
            dropout=0,
            # res           1, 2, 4, 8,16,32,64,128,256,512, 1K, 2K
            channel_mult=  (1,1.5,2, 3, 4, 6, 8, 12, 16, 24, 32, 48),
            num_res_blocks=(1, 1, 1, 1, 1, 2, 2, 2,   2,  2,  2,  2),
            # spec_cond:    1, 0, 0, 1, 0, 0, 1, 0,   0,  1,  0,  0)
            # attn:         0, 0, 0, 0, 0, 0, 0, 0,   0,  1,  1,  1
            attention_resolutions=(512,1024,2048),
            conv_resample=True,
            dims=1,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            kernel_size=3,
            scale_factor=2,
            time_embed_dim_multiplier=4,
            efficient_convs=True,  # Uses kernels with width of 1 in several places rather than 3.
            use_scale_shift_norm=True,
            freeze_main=False,
            # Parameters for regularization.
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dims = dims
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.alignment_size = 2 ** (len(channel_mult)+1)
        self.in_mel_channels = in_mel_channels
        padding = 1 if kernel_size == 3 else 2
        down_kernel = 1 if efficient_convs else 3

        time_embed_dim = model_channels * time_embed_dim_multiplier
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        conditioning_dim = model_channels * conditioning_dim_factor
        self.structural_cond_input = nn.Conv1d(in_mel_channels, conditioning_dim, 3, padding=1)
        self.aligned_latent_padding_embedding = nn.Parameter(torch.randn(1,in_mel_channels,1))
        self.unconditioned_embedding = nn.Parameter(torch.randn(1,conditioning_dim,1))
        self.structural_processor = StructuralProcessor(conditioning_dim, dropout)
        self.surrogate_head = nn.Conv1d(conditioning_dim//16, in_channels, 1)

        self.input_block = conv_nd(dims, in_channels, model_channels//2, kernel_size, padding=padding)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, model_channels, model_channels, kernel_size, padding=padding)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, (mult, num_blocks) in enumerate(zip(channel_mult, num_res_blocks)):
            for _ in range(num_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        kernel_size=kernel_size,
                        efficient_config=efficient_convs,
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
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, factor=scale_factor, ksize=down_kernel, pad=0 if down_kernel == 1 else 1
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
                kernel_size=kernel_size,
                efficient_config=efficient_convs,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                kernel_size=kernel_size,
                efficient_config=efficient_convs,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, (mult, num_blocks) in list(enumerate(zip(channel_mult, num_res_blocks)))[::-1]:
            for i in range(num_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        kernel_size=kernel_size,
                        efficient_config=efficient_convs,
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
                        )
                    )
                if level and i == num_blocks:
                    out_ch = ch
                    layers.append(
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, factor=scale_factor)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, kernel_size, padding=padding)),
        )

        if freeze_main:
            for p in self.parameters():
                p.DO_NOT_TRAIN = True
                p.requires_grad = False
            for m in [self.structural_processor, self.structural_cond_input, self.surrogate_head]:
                for p in m.parameters():
                    del p.DO_NOT_TRAIN
                    p.requires_grad = True


    def get_grad_norm_parameter_groups(self):
        if self.freeze_main_net:
            return {}
        groups = {
            'input_blocks': list(self.input_blocks.parameters()),
            'output_blocks': list(self.output_blocks.parameters()),
            'middle_transformer': list(self.middle_block.parameters()),
            'structural_processor': list(self.structural_processor.parameters()),
        }
        return groups

    def is_latent(self, t):
        return t.shape[1] != self.in_mel_channels

    def fix_alignment(self, x, aligned_conditioning):
        """
        The UNet requires that the input <x> is a certain multiple of 2, defined by the UNet depth. Enforce this by
        padding both <x> and <aligned_conditioning> before forward propagation and removing the padding before returning.
        """
        cm = ceil_multiple(x.shape[-1], self.alignment_size)
        if cm != 0:
            pc = (cm-x.shape[-1])/x.shape[-1]
            x = F.pad(x, (0,cm-x.shape[-1]))
            # Also fix aligned_latent, which is aligned to x.
            if self.is_latent(aligned_conditioning):
                aligned_conditioning = torch.cat([aligned_conditioning,
                                                  self.aligned_latent_padding_embedding.repeat(x.shape[0], 1, int(pc * aligned_conditioning.shape[-1]))], dim=-1)
            else:
                aligned_conditioning = F.pad(aligned_conditioning, (0,int(pc*aligned_conditioning.shape[-1])))
        return x, aligned_conditioning

    def forward(self, x, timesteps, aligned_conditioning, conditioning_free=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param aligned_conditioning: an aligned latent or sequence of tokens providing useful data about the sample to be produced.
        :param conditioning_free: When set, all conditioning inputs (including tokens and conditioning_input) will not be considered.
        :return: an [N x C x ...] Tensor of outputs.
        """

        # Shuffle aligned_latent to BxCxS format
        if self.is_latent(aligned_conditioning):
            aligned_conditioning = aligned_conditioning.permute(0, 2, 1)

        # Fix input size to the proper multiple of 2 so we don't get alignment errors going down and back up the U-net.
        orig_x_shape = x.shape[-1]
        x, aligned_conditioning = self.fix_alignment(x, aligned_conditioning)

        with autocast(x.device.type, enabled=self.enable_fp16):
            hs = []
            time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

            # Note: this block does not need to repeated on inference, since it is not timestep-dependent.
            if conditioning_free:
                code_emb = self.unconditioned_embedding.repeat(x.shape[0], 1, 1)
                surrogate = torch.zeros_like(x)
            else:
                code_emb = self.structural_cond_input(aligned_conditioning)
                code_emb = self.structural_processor(code_emb)
                code_emb = F.interpolate(code_emb, size=(x.shape[-1],), mode='linear')
                surrogate = self.surrogate_head(code_emb)

            # Everything after this comment is timestep dependent.
            x = self.input_block(x)
            x = torch.cat([x, code_emb], dim=1)

            time_emb = time_emb.float()
            h = x
            for k, module in enumerate(self.input_blocks):
                with autocast(x.device.type, enabled=self.enable_fp16 and not first):
                    # First block has autocast disabled to allow a high precision signal to be properly vectorized.
                    h = module(h, time_emb)
                hs.append(h)
            h = self.middle_block(h, time_emb)
            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, time_emb)

        # Last block also has autocast disabled for high-precision outputs.
        h = h.float()
        out = self.out(h)

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        params = [self.aligned_latent_padding_embedding, self.unconditioned_embedding]
        for p in params:
            extraneous_addition = extraneous_addition + p.mean()
        out = out + extraneous_addition * 0

        return out[:, :, :orig_x_shape], surrogate[:, :, :orig_x_shape]


@register_model
def register_unet_diffusion_waveform_gen2(opt_net, opt):
    return DiffusionTts(**opt_net['kwargs'])


if __name__ == '__main__':
    clip = torch.randn(2, 1, 32868)
    aligned_sequence = torch.randn(2,120,128)
    ts = torch.LongTensor([600, 600])
    model = DiffusionTts(128,
                         channel_mult=[1,1.5,2, 3, 4, 6, 8],
                         num_res_blocks=[2, 2, 2, 2, 2, 2, 1],
                         attention_resolutions=[],
                         num_heads=8,
                         kernel_size=3,
                         scale_factor=2,
                         time_embed_dim_multiplier=4,
                         efficient_convs=False)
    # Test with sequence aligned conditioning
    o = model(clip, ts, aligned_sequence)

