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
from models.lucidrains.x_transformers import Encoder, Attention
from scripts.audio.gen.use_mel2vec_codes import collapse_codegroups
from trainer.injectors.audio_injectors import normalize_mel
from trainer.networks import register_model
from utils.util import checkpoint


def is_latent(t):
    return t.dtype == torch.float

def is_sequence(t):
    return t.dtype == torch.long


class TransformerDiffusion(nn.Module):
    """
    A diffusion model composed entirely of stacks of transformer layers. Why would you do it any other way?
    """

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
            # Parameters for regularization.
            layer_drop=.1,
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.layer_drop = layer_drop
        heads = model_channels//64

        self.inp_block = conv_nd(1, in_channels, model_channels, 3, 1, 1)

        self.time_embed = nn.Sequential(
            linear(model_channels, model_channels),
            nn.SiLU(),
            linear(model_channels, model_channels),
        )
        self.conditioning_embedder = nn.Sequential(nn.Conv1d(in_channels, model_channels // 2, 3, padding=1, stride=2),
                                                   nn.Conv1d(model_channels//2, model_channels,3,padding=1,stride=2))
        self.conditioning_encoder = Encoder(
                    dim=model_channels,
                    depth=4,
                    heads=heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                )

        # Either code_converter or latent_converter is used, depending on what type of conditioning data is fed.
        # This model is meant to be able to be trained on both for efficiency purposes - it is far less computationally
        # complex to generate tokens, while generating latents will normally mean propagating through a deep autoregressive
        # transformer network.
        self.embeddings = nn.ModuleList([nn.Embedding(in_vectors, model_channels//in_groups) for _ in range(in_groups)])
        self.latent_conditioner = nn.Sequential(
            nn.Conv1d(in_latent_channels, model_channels, 3, padding=1),
            Encoder(
                    dim=model_channels,
                    depth=2,
                    heads=heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                )
        )
        self.latent_fade = nn.Parameter(torch.zeros(1,1,model_channels))
        self.code_converter = Encoder(
                    dim=model_channels,
                    depth=3,
                    heads=heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                )

        self.unconditioned_embedding = nn.Parameter(torch.randn(1,1,model_channels))
        self.integrator = nn.Linear(model_channels * 2, model_channels)
        self.mel_head = nn.Conv1d(model_channels, in_channels, kernel_size=3, padding=1)

        self.layers = Encoder(
                    dim=model_channels,
                    depth=num_layers,
                    heads=heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rms_scaleshift_norm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                    zero_init_branch_output=True,
                )

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(conv_nd(1, model_channels, out_channels, 3, padding=1)),
        )

        self.debug_codes = {}

    def get_grad_norm_parameter_groups(self):
        groups = {
            'contextual_embedder': list(self.conditioning_embedder.parameters()),
            'layers': list(self.layers.parameters()) + list(self.integrator.parameters()) + list(self.inp_block.parameters()),
            'code_converters': list(self.embeddings.parameters()) + list(self.code_converter.parameters()) + list(self.latent_conditioner.parameters()),
            'time_embed': list(self.time_embed.parameters()),
        }
        return groups

    def timestep_independent(self, codes, conditioning_input, expected_seq_len, prenet_latent=None, return_code_pred=False):
        cond_emb = self.conditioning_embedder(conditioning_input).permute(0,2,1)
        cond_emb = self.conditioning_encoder(cond_emb)[:, 0]

        code_emb = [embedding(codes[:, :, i]) for i, embedding in enumerate(self.embeddings)]
        code_emb = torch.cat(code_emb, dim=-1)
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

        expanded_code_emb = F.interpolate(code_emb.permute(0,2,1), size=expected_seq_len, mode='nearest').permute(0,2,1)
        if not return_code_pred:
            return expanded_code_emb, cond_emb
        else:
            # Perform the mel_head computation on the pre-exanded code embeddings, then interpolate it separately.
            mel_pred = self.mel_head(code_emb.permute(0,2,1))
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
        if not return_code_pred:
            unused_params.extend(list(self.mel_head.parameters()))
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
        x = self.inp_block(x).permute(0,2,1)
        x = torch.cat([x, code_emb], dim=2)
        x = self.integrator(x)
        x = self.layers(x, norm_scale_shift_inp=blk_emb)

        x = x.float().permute(0,2,1)
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
def register_transformer_diffusion(opt_net, opt):
    return TransformerDiffusion(**opt_net['kwargs'])


if __name__ == '__main__':
    clip = torch.randn(2, 256, 400)
    aligned_latent = torch.randn(2,100,512)
    aligned_sequence = torch.randint(0,8,(2,100,8))
    cond = torch.randn(2, 256, 400)
    ts = torch.LongTensor([600, 600])
    model = TransformerDiffusion(512, layer_drop=.3, unconditioned_percentage=.5)
    o = model(clip, ts, aligned_sequence, cond, return_code_pred=True)
    #o = model(clip, ts, aligned_sequence, cond, aligned_latent)

