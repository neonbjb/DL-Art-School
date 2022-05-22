import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import autocast

from models.arch_util import ResBlock
from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import AttentionBlock, TimestepEmbedSequential, TimestepBlock
from scripts.audio.gen.use_mel2vec_codes import collapse_codegroups
from trainer.injectors.audio_injectors import normalize_mel
from trainer.networks import register_model
from utils.util import checkpoint


def is_latent(t):
    return t.dtype == torch.float

def is_sequence(t):
    return t.dtype == torch.long


class TimestepResBlock(TimestepBlock):
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


class DiffusionLayer(TimestepBlock):
    def __init__(self, model_channels, dropout, num_heads):
        super().__init__()
        self.resblk = TimestepResBlock(model_channels, model_channels, dropout, model_channels, dims=1, use_scale_shift_norm=True)
        self.attn = AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True)

    def forward(self, x, time_emb):
        y = self.resblk(x, time_emb)
        return self.attn(y)


class NonTimestepResidualAttentionNorm(nn.Module):
    def __init__(self, model_channels, dropout):
        super().__init__()
        self.resblk = ResBlock(dims=1, channels=model_channels, dropout=dropout)
        self.attn = AttentionBlock(model_channels, num_heads=model_channels//64, relative_pos_embeddings=True)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=model_channels)

    def forward(self, x):
        h = self.resblk(x)
        h = self.norm(h)
        h = self.attn(h)
        return h


class FlatDiffusion(nn.Module):
    def __init__(
            self,
            model_channels=512,
            num_layers=8,
            in_channels=256,
            in_latent_channels=512,
            in_vectors=8,
            in_groups=8,
            out_channels=512,  # mean and variance
            dropout=0,
            use_fp16=False,
            num_heads=8,
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

        self.inp_block = conv_nd(1, in_channels, model_channels, 3, 1, 1)
        # TODO: I'd really like to see if this could be ablated. It seems useless to me: why can't the embedding just learn this mapping directly?
        self.time_embed = nn.Sequential(
            linear(model_channels, model_channels),
            nn.SiLU(),
            linear(model_channels, model_channels),
        )

        # Either code_converter or latent_converter is used, depending on what type of conditioning data is fed.
        # This model is meant to be able to be trained on both for efficiency purposes - it is far less computationally
        # complex to generate tokens, while generating latents will normally mean propagating through a deep autoregressive
        # transformer network.
        self.embeddings = nn.ModuleList([nn.Embedding(in_vectors, model_channels//in_groups) for _ in range(in_groups)])
        self.latent_conditioner = nn.Sequential(
            nn.Conv1d(in_latent_channels, model_channels, 3, padding=1),
            NonTimestepResidualAttentionNorm(model_channels, dropout),
            NonTimestepResidualAttentionNorm(model_channels, dropout),
            nn.Conv1d(model_channels, model_channels, 3, padding=1),
        )
        self.latent_fade = nn.Parameter(torch.zeros(1,model_channels,1))
        self.code_converter = nn.Sequential(
            NonTimestepResidualAttentionNorm(model_channels, dropout),
            NonTimestepResidualAttentionNorm(model_channels, dropout),
            NonTimestepResidualAttentionNorm(model_channels, dropout),
            nn.Conv1d(model_channels, model_channels, 3, padding=1),
        )
        self.conditioning_embedder = nn.Sequential(nn.Conv1d(in_channels, model_channels // 2, 3, padding=1, stride=2),
                                                   nn.Conv1d(model_channels//2, model_channels,3,padding=1,stride=2),
                                                   NonTimestepResidualAttentionNorm(model_channels, dropout),
                                                   NonTimestepResidualAttentionNorm(model_channels, dropout),
                                                   NonTimestepResidualAttentionNorm(model_channels, dropout))
        self.unconditioned_embedding = nn.Parameter(torch.randn(1,model_channels,1))
        self.integrating_conv = nn.Conv1d(model_channels*2, model_channels, kernel_size=1)
        self.mel_head = nn.Conv1d(model_channels, in_channels, kernel_size=3, padding=1)

        self.layers = nn.ModuleList([DiffusionLayer(model_channels, dropout, num_heads) for _ in range(num_layers)] +
                                    [TimestepResBlock(model_channels, model_channels, dropout, dims=1, use_scale_shift_norm=True) for _ in range(3)])

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(conv_nd(1, model_channels, out_channels, 3, padding=1)),
        )

        self.debug_codes = {}

    def get_grad_norm_parameter_groups(self):
        groups = {
            'contextual_embedder': list(self.conditioning_embedder.parameters()),
            'layers': list(self.layers.parameters()) + list(self.integrating_conv.parameters()) + list(self.inp_block.parameters()),
            'code_converters': list(self.embeddings.parameters()) + list(self.code_converter.parameters()) + list(self.latent_conditioner.parameters()),
            'time_embed': list(self.time_embed.parameters()),
        }
        return groups

    def timestep_independent(self, codes, conditioning_input, expected_seq_len, prenet_latent=None, return_code_pred=False):
        cond_emb = self.conditioning_embedder(conditioning_input)[:, :, 0]

        # Shuffle prenet_latent to BxCxS format
        if prenet_latent is not None:
            prenet_latent = prenet_latent.permute(0, 2, 1)

        code_emb = [embedding(codes[:, :, i]) for i, embedding in enumerate(self.embeddings)]
        code_emb = torch.cat(code_emb, dim=-1).permute(0,2,1)
        if prenet_latent is not None:
            latent_conditioning = self.latent_conditioner(prenet_latent)
            code_emb = code_emb + latent_conditioning * self.latent_fade

        unconditioned_batches = torch.zeros((code_emb.shape[0], 1, 1), device=code_emb.device)
        # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
        if self.training and self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand((code_emb.shape[0], 1, 1),
                                               device=code_emb.device) < self.unconditioned_percentage
            code_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(codes.shape[0], 1, 1),
                                   code_emb)
        code_emb = self.code_converter(code_emb)

        expanded_code_emb = F.interpolate(code_emb, size=expected_seq_len, mode='nearest')
        if not return_code_pred:
            return expanded_code_emb, cond_emb
        else:
            # Perform the mel_head computation on the pre-exanded code embeddings, then interpolate it separately.
            mel_pred = self.mel_head(code_emb)
            mel_pred = F.interpolate(mel_pred, size=expected_seq_len, mode='nearest')
            # Multiply mel_pred by !unconditioned_branches, which drops the gradient on unconditioned branches.
            # This is because we don't want that gradient being used to train parameters through the codes_embedder as
            # it unbalances contributions to that network from the MSE loss.
            mel_pred = mel_pred * unconditioned_batches.logical_not()
            return expanded_code_emb, cond_emb, mel_pred


    def forward(self, x, timesteps,
                codes=None, conditioning_input=None, prenet_latent=None,
                precomputed_code_embeddings=None, precomputed_cond_embeddings=None,
                conditioning_free=False, return_code_pred=False):
        """
        Apply the model to an input batch.

        There are two ways to call this method:
        1) Specify codes, conditioning_input and optionally prenet_latent
        2) Specify precomputed_code_embeddings and precomputed_cond_embeddings, retrieved by calling timestep_independent yourself.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param codes: an aligned latent or sequence of tokens providing useful data about the sample to be produced.
        :param conditioning_input: a full-resolution audio clip that is used as a reference to the style you want decoded.
        :param prenet_latent: optional latent vector aligned with codes derived from a prior network.
        :param precomputed_code_embeddings: Code embeddings returned from self.timestep_independent()
        :param precomputed_cond_embeddings: Conditional embeddings returned from self.timestep_independent()
        :param conditioning_free: When set, all conditioning inputs (including tokens and conditioning_input) will not be considered.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if precomputed_code_embeddings is not None:
            assert precomputed_cond_embeddings is not None, "Must specify both precomputed embeddings if one is specified"
            assert codes is None and conditioning_input is None and prenet_latent is None, "Do not provide precomputed embeddings and the other parameters. It is unclear what you want me to do here."
        assert not (return_code_pred and precomputed_code_embeddings is not None), "I cannot compute a code_pred output for you."

        unused_params = []
        if conditioning_free:
            code_emb = self.unconditioned_embedding.repeat(x.shape[0], 1, x.shape[-1])
            unused_params.extend(list(self.code_converter.parameters()) + list(self.code_embedding.parameters()))
            unused_params.extend(list(self.latent_conditioner.parameters()))
        else:
            if precomputed_code_embeddings is not None:
                code_emb = precomputed_code_embeddings
                cond_emb = precomputed_cond_embeddings
            else:
                code_emb, cond_emb, mel_pred = self.timestep_independent(codes, conditioning_input, x.shape[-1], prenet_latent, True)
                if prenet_latent is None:
                    unused_params.extend(list(self.latent_conditioner.parameters()) + [self.latent_fade])
            unused_params.append(self.unconditioned_embedding)

        blk_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)) + cond_emb
        x = self.inp_block(x)
        x = torch.cat([x, code_emb], dim=1)
        x = self.integrating_conv(x)
        for i, lyr in enumerate(self.layers):
            # Do layer drop where applicable. Do not drop first and last layers.
            if self.training and self.layer_drop > 0 and i != 0 and i != (len(self.layers)-1) and random.random() < self.layer_drop:
                unused_params.extend(list(lyr.parameters()))
            else:
                # First and last blocks will have autocast disabled for improved precision.
                with autocast(x.device.type, enabled=self.enable_fp16 and i != 0):
                    x = lyr(x, blk_emb)

        x = x.float()
        out = self.out(x)

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        out = out + extraneous_addition * 0

        if return_code_pred:
            return out, mel_pred
        return out

    def get_conditioning_latent(self, conditioning_input):
        speech_conditioning_input = conditioning_input.unsqueeze(1) if len(
            conditioning_input.shape) == 3 else conditioning_input
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(self.conditioning_embedder(speech_conditioning_input[:, j]))
        conds = torch.cat(conds, dim=-1)
        return conds.mean(dim=-1)

@register_model
def register_flat_diffusion(opt_net, opt):
    return FlatDiffusion(**opt_net['kwargs'])


if __name__ == '__main__':
    clip = torch.randn(2, 256, 400)
    aligned_latent = torch.randn(2,100,512)
    aligned_sequence = torch.randint(0,8,(2,100,8))
    cond = torch.randn(2, 256, 400)
    ts = torch.LongTensor([600, 600])
    model = FlatDiffusion(512, layer_drop=.3, unconditioned_percentage=.5)
    o = model(clip, ts, aligned_sequence, cond, return_code_pred=True)
    o = model(clip, ts, aligned_sequence, cond, aligned_latent)

