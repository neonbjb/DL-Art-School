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


def is_latent(t):
    return t.dtype == torch.float

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


class DiffusionLayer(nn.Module):
    def __init__(self, model_channels, dropout, num_heads):
        super().__init__()
        self.resblk = ResBlock(model_channels, model_channels, dropout, model_channels, dims=1, use_scale_shift_norm=True)
        self.attn = AttentionBlock(model_channels, num_heads)

    def forward(self, x, time_emb):
        y = self.resblk(x, time_emb)
        return self.attn(y)


class DiffusionTtsFlat(nn.Module):
    def __init__(
            self,
            model_channels=512,
            num_layers=8,
            in_channels=100,
            in_latent_channels=512,
            in_tokens=8193,
            max_timesteps=4000,
            max_positions=4000,
            out_channels=200,  # mean and variance
            dropout=0,
            use_fp16=False,
            num_heads=16,
            # Parameters for regularization.
            layer_drop=.1,
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_heads = num_heads
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.layer_drop = layer_drop

        self.inp_block = conv_nd(1, in_channels, model_channels//2, 3, 1, 1)
        self.position_embed = nn.Embedding(max_positions, model_channels//2)
        self.time_embed = nn.Embedding(max_timesteps, model_channels)

        # Either code_converter or latent_converter is used, depending on what type of conditioning data is fed.
        # This model is meant to be able to be trained on both for efficiency purposes - it is far less computationally
        # complex to generate tokens, while generating latents will normally mean propagating through a deep autoregressive
        # transformer network.
        self.code_converter = nn.Sequential(
            nn.Embedding(in_tokens, model_channels),
            CheckpointedXTransformerEncoder(
                needs_permute=False,
                max_seq_len=-1,
                use_pos_emb=False,
                attn_layers=Encoder(
                    dim=model_channels,
                    depth=3,
                    heads=num_heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_emb_dim=True,
                )
            )
        )
        self.latent_converter = nn.Conv1d(in_latent_channels, model_channels, 1)
        if in_channels > 60:  # It's a spectrogram.
            self.contextual_embedder = nn.Sequential(nn.Conv1d(in_channels,model_channels,3,padding=1,stride=2),
                                                     CheckpointedXTransformerEncoder(
                                                         needs_permute=True,
                                                         max_seq_len=-1,
                                                         use_pos_emb=False,
                                                         attn_layers=Encoder(
                                                             dim=model_channels,
                                                             depth=4,
                                                             heads=num_heads,
                                                             ff_dropout=dropout,
                                                             attn_dropout=dropout,
                                                             use_rmsnorm=True,
                                                             ff_glu=True,
                                                             rotary_emb_dim=True,
                                                         )
                                                     ))
        else:
            self.contextual_embedder = AudioMiniEncoder(1, model_channels, base_channels=32, depth=6, resnet_blocks=1,
                                                        attn_blocks=3, num_attn_heads=8, dropout=dropout, downsample_factor=4, kernel_size=5)
        self.conditioning_conv = nn.Conv1d(model_channels*2, model_channels, 1)
        self.unconditioned_embedding = nn.Parameter(torch.randn(1,model_channels,1))
        self.conditioning_timestep_integrator = TimestepEmbedSequential(
            ResBlock(model_channels, model_channels, dropout, out_channels=model_channels, dims=1, kernel_size=1, use_scale_shift_norm=True),
            AttentionBlock(model_channels, num_heads=num_heads),
            ResBlock(model_channels, model_channels, dropout, out_channels=model_channels, dims=1, kernel_size=1, use_scale_shift_norm=True),
            AttentionBlock(model_channels, num_heads=num_heads),
            ResBlock(model_channels, model_channels, dropout, out_channels=model_channels//2, dims=1, kernel_size=1, use_scale_shift_norm=True),
        )

        self.layers = nn.ModuleList([DiffusionLayer(model_channels, dropout, num_heads) for _ in range(num_layers)])

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(conv_nd(1, model_channels, out_channels, 3, padding=1)),
        )

    def get_grad_norm_parameter_groups(self):
        groups = {
            'minicoder': list(self.contextual_embedder.parameters()),
            'layers': list(self.layers),
        }
        return groups

    def forward(self, x, timesteps, aligned_conditioning, conditioning_input, lr_input=None, conditioning_free=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param aligned_conditioning: an aligned latent or sequence of tokens providing useful data about the sample to be produced.
        :param conditioning_input: a full-resolution audio clip that is used as a reference to the style you want decoded.
        :param lr_input: for super-sampling models, a guidance audio clip at a lower sampling rate.
        :param conditioning_free: When set, all conditioning inputs (including tokens and conditioning_input) will not be considered.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # Shuffle aligned_latent to BxCxS format
        if is_latent(aligned_conditioning):
            aligned_conditioning = aligned_conditioning.permute(0, 2, 1)

        # Note: this block does not need to repeated on inference, since it is not timestep-dependent or x-dependent.
        unused_params = []
        if conditioning_free:
            code_emb = self.unconditioned_embedding.repeat(x.shape[0], 1, 1)
        else:
            unused_params.append(self.unconditioned_embedding)
            cond_emb = self.contextual_embedder(conditioning_input)
            if len(cond_emb.shape) == 3:  # Just take the first element.
                cond_emb = cond_emb[:, :, 0]
            if is_latent(aligned_conditioning):
                code_emb = self.latent_converter(aligned_conditioning)
                unused_params.extend(list(self.code_converter.parameters()))
            else:
                code_emb = self.code_converter(aligned_conditioning)
                unused_params.extend(list(self.latent_converter.parameters()))
            cond_emb_spread = cond_emb.unsqueeze(-1).repeat(1, 1, code_emb.shape[-1])
            code_emb = self.conditioning_conv(torch.cat([cond_emb_spread, code_emb], dim=1))
        # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
        if self.training and self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand((code_emb.shape[0], 1, 1),
                                               device=code_emb.device) < self.unconditioned_percentage
            code_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(x.shape[0], 1, 1),
                                   code_emb)

        # Everything after this comment is timestep dependent.
        time_emb = self.time_embed(timesteps)
        code_emb = self.conditioning_timestep_integrator(code_emb, time_emb)
        pos_emb = self.position_embed(torch.arange(0, x.shape[-1], device=x.device)).unsqueeze(0).repeat(x.shape[0],1,1).permute(0,2,1)
        x = self.inp_block(x) + pos_emb
        x = torch.cat([x, F.interpolate(code_emb, size=x.shape[-1], mode='nearest')], dim=1)
        for i, lyr in enumerate(self.layers):
            # Do layer drop where applicable. Do not drop first and last layers.
            if self.training and self.layer_drop > 0 and i != 0 and i != (len(self.layers)-1) and random.random() < self.layer_drop:
                unused_params.extend(list(lyr.parameters()))
            else:
                # First and last blocks will have autocast disabled for improved precision.
                with autocast(x.device.type, enabled=self.enable_fp16 and i != 0):
                    x = lyr(x, time_emb)

        x = x.float()
        out = self.out(x)

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        out = out + extraneous_addition * 0

        return out


@register_model
def register_diffusion_tts_flat(opt_net, opt):
    return DiffusionTtsFlat(**opt_net['kwargs'])


if __name__ == '__main__':
    clip = torch.randn(2, 100, 400)
    aligned_latent = torch.randn(2,388,512)
    aligned_sequence = torch.randint(0,8192,(2,388))
    cond = torch.randn(2, 100, 400)
    ts = torch.LongTensor([600, 600])
    model = DiffusionTtsFlat(512, layer_drop=.3)
    # Test with latent aligned conditioning
    o = model(clip, ts, aligned_latent, cond)
    # Test with sequence aligned conditioning
    o = model(clip, ts, aligned_sequence, cond)

