import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torchaudio.transforms import TimeMasking, FrequencyMasking

from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import AttentionBlock, TimestepEmbedSequential, TimestepBlock
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


class DiffusionLayer(TimestepBlock):
    def __init__(self, model_channels, dropout, num_heads):
        super().__init__()
        self.resblk = ResBlock(model_channels, model_channels, dropout, model_channels, dims=1, use_scale_shift_norm=True)
        self.attn = AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True)

    def forward(self, x, time_emb):
        y = self.resblk(x, time_emb)
        return self.attn(y)


class MusicGenerator(nn.Module):
    def __init__(
            self,
            model_channels=512,
            num_layers=8,
            in_channels=100,
            out_channels=200,  # mean and variance
            dropout=0,
            use_fp16=False,
            num_heads=16,
            # Parameters for regularization.
            layer_drop=.1,
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
            # Masking parameters.
            frequency_mask_percent_max=0,
            time_mask_percent_max=0,
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
        self.time_mask_percent_max = time_mask_percent_max
        self.frequency_mask_percent_mask = frequency_mask_percent_max

        self.inp_block = conv_nd(1, in_channels, model_channels, 3, 1, 1)
        self.time_embed = nn.Sequential(
            linear(model_channels, model_channels),
            nn.SiLU(),
            linear(model_channels, model_channels),
        )

        self.conditioner = nn.Sequential(
            nn.Conv1d(in_channels, model_channels, 3, padding=1),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
        )
        self.unconditioned_embedding = nn.Parameter(torch.randn(1,model_channels,1))
        self.conditioning_timestep_integrator = TimestepEmbedSequential(
            DiffusionLayer(model_channels, dropout, num_heads),
            DiffusionLayer(model_channels, dropout, num_heads),
        )
        self.integrating_conv = nn.Conv1d(model_channels*2, model_channels, kernel_size=1)
        self.layers = nn.ModuleList([DiffusionLayer(model_channels, dropout, num_heads) for _ in range(num_layers)] +
                                    [ResBlock(model_channels, model_channels, dropout, dims=1, use_scale_shift_norm=True) for _ in range(3)])

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(conv_nd(1, model_channels, out_channels, 3, padding=1)),
        )

    def get_grad_norm_parameter_groups(self):
        groups = {
            'layers': list(self.layers.parameters()),
            'conditioner': list(self.conditioner.parameters()) + list(self.conditioner.parameters()),
            'timestep_integrator': list(self.conditioning_timestep_integrator.parameters()) + list(self.integrating_conv.parameters()),
            'time_embed': list(self.time_embed.parameters()),
        }
        return groups

    def do_masking(self, truth):
        b, c, s = truth.shape
        mask = torch.ones_like(truth)
        if self.frequency_mask_percent_mask > 0:
            # Frequency mask
            cs = random.randint(0, c-10)
            ce = min(c-1, cs+random.randint(1, int(self.frequency_mask_percent_mask*c)))
            mask[:, cs:ce] = 0
        else:
            # Time mask
            cs = random.randint(0, s-5)
            ce = min(s-1, cs+random.randint(1, int(self.frequency_mask_percent_mask*s)))
            mask[:, :, cs:ce] = 0
        return truth * mask


    def timestep_independent(self, truth, expected_seq_len, return_code_pred):
        truth_emb = self.conditioner(truth)
        # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
        if self.training and self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand((truth_emb.shape[0], 1, 1),
                                               device=truth_emb.device) < self.unconditioned_percentage
            truth_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(truth.shape[0], 1, 1),
                                   truth_emb)
        return truth_emb


    def forward(self, x, timesteps, truth=None, precomputed_aligned_embeddings=None, conditioning_free=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param truth: Input value is either pre-masked (in inference), or unmasked (during training)
        :param precomputed_aligned_embeddings: Embeddings returned from self.timestep_independent()
        :param conditioning_free: When set, all conditioning inputs (including tokens and conditioning_input) will not be considered.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert precomputed_aligned_embeddings is not None or truth is not None

        unused_params = []
        if conditioning_free:
            truth_emb = self.unconditioned_embedding.repeat(x.shape[0], 1, x.shape[-1])
            unused_params.extend(list(self.conditioner.parameters()))
        else:
            if precomputed_aligned_embeddings is not None:
                truth_emb = precomputed_aligned_embeddings
            else:
                if self.training:
                    truth = self.do_masking(truth)
                truth_emb = self.timestep_independent(truth, x.shape[-1], True)
            unused_params.append(self.unconditioned_embedding)

        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        truth_emb = self.conditioning_timestep_integrator(truth_emb, time_emb)
        x = self.inp_block(x)
        x = torch.cat([x, truth_emb], dim=1)
        x = self.integrating_conv(x)
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
def register_music_gap_gen(opt_net, opt):
    return MusicGenerator(**opt_net['kwargs'])


if __name__ == '__main__':
    clip = torch.randn(2, 100, 400)
    aligned_latent = torch.randn(2,100,388)
    ts = torch.LongTensor([600, 600])
    model = MusicGenerator(512, layer_drop=.3, unconditioned_percentage=.5)
    o = model(clip, ts, aligned_latent)

