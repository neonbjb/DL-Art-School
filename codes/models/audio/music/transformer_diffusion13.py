import itertools
from random import randrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils

from models.arch_util import ResBlock, TimestepEmbedSequential, AttentionBlock, build_local_attention_mask
from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import TimestepBlock
from trainer.networks import register_model
from utils.util import checkpoint


def is_latent(t):
    return t.dtype == torch.float


def is_sequence(t):
    return t.dtype == torch.long


class SubBlock(nn.Module):
    def __init__(self, inp_dim, contraction_dim, blk_dim, heads, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.blk_emb_proj = nn.Conv1d(blk_dim, inp_dim, 1)
        self.attn = AttentionBlock(inp_dim, out_channels=contraction_dim, num_heads=heads)
        self.attnorm = nn.GroupNorm(8, contraction_dim)
        self.ff = nn.Conv1d(inp_dim+contraction_dim, contraction_dim, kernel_size=3, padding=1)
        self.ffnorm = nn.GroupNorm(8, contraction_dim)
        self.mask = build_local_attention_mask(n=4000, l=64, fixed_region=8)
        self.mask_initialized = False

    def forward(self, x, blk_emb):
        if self.mask is not None and not self.mask_initialized:
            self.mask = self.mask.to(x.device)
            self.mask_initialized = True
        blk_enc = self.blk_emb_proj(blk_emb)
        ah = self.dropout(self.attn(torch.cat([blk_enc, x], dim=-1), mask=self.mask))
        ah = ah[:,:,blk_enc.shape[-1]:]  # Strip off the blk_emc used for attention and re-align with x.
        ah = F.gelu(self.attnorm(ah))
        h = torch.cat([ah, x], dim=1)
        hf = self.dropout(checkpoint(self.ff, h))
        hf = F.gelu(self.ffnorm(hf))
        h = torch.cat([h, hf], dim=1)
        return h


class ConcatAttentionBlock(TimestepBlock):
    def __init__(self, trunk_dim, contraction_dim, heads, dropout):
        super().__init__()
        self.prenorm = nn.GroupNorm(8, trunk_dim)
        self.block1 = SubBlock(trunk_dim, contraction_dim, trunk_dim, heads, dropout)
        self.block2 = SubBlock(trunk_dim+contraction_dim*2, contraction_dim, trunk_dim, heads, dropout)
        self.out = nn.Conv1d(contraction_dim*4, trunk_dim, kernel_size=1, bias=False)
        self.out.weight.data.zero_()

    def forward(self, x, blk_emb):
        h = self.prenorm(x)
        h = self.block1(h, blk_emb)
        h = self.block2(h, blk_emb)
        h = self.out(h[:,x.shape[1]:])
        return h + x


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 num_resolutions,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=5, stride=2)
        self.resolution_embedding = nn.Embedding(num_resolutions, embedding_dim)
        self.resolution_embedding.weight.data.mul(.1)  # Reduces the relative influence of this embedding from the start.
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads, do_checkpoint=do_checkpointing))
            attn.append(ResBlock(embedding_dim, dims=1, checkpointing_enabled=do_checkpointing))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing

    def forward(self, x, resolution):
        h = self.init(x) + self.resolution_embedding(resolution).unsqueeze(-1)
        h = self.attn(h)
        return h[:, :, :6]


class TransformerDiffusion(nn.Module):
    """
    A diffusion model composed entirely of stacks of transformer layers. Why would you do it any other way?
    """
    def __init__(
            self,
            time_embed_dim=256,
            resolution_steps=8,
            max_window=384,
            model_channels=1024,
            contraction_dim=256,
            num_layers=8,
            in_channels=256,
            input_vec_dim=1024,
            out_channels=512,  # mean and variance
            num_heads=4,
            dropout=0,
            use_fp16=False,
            # Parameters for regularization.
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.time_embed_dim = time_embed_dim
        self.out_channels = out_channels
        self.dropout = dropout
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.resolution_steps = resolution_steps
        self.max_window = max_window
        self.preprocessed = None

        self.time_embed = nn.Sequential(
            linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, model_channels),
        )
        self.resolution_embed = nn.Embedding(resolution_steps, model_channels)
        self.conditioning_encoder = ConditioningEncoder(in_channels, model_channels, resolution_steps, num_attn_heads=model_channels//64)
        self.unconditioned_embedding = nn.Parameter(torch.randn(1,model_channels,6))
        self.unconditioned_prior = nn.Parameter(torch.zeros(1,in_channels,1))

        self.inp_block = conv_nd(1, in_channels+input_vec_dim, model_channels, 3, 1, 1)
        self.layers = TimestepEmbedSequential(*[ConcatAttentionBlock(model_channels, contraction_dim, num_heads, dropout) for _ in range(num_layers)])

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(conv_nd(1, model_channels, out_channels, 3, padding=1)),
        )

        self.debug_codes = {}

    def get_grad_norm_parameter_groups(self):
        attn1 = list(itertools.chain.from_iterable([lyr.block1.attn.parameters() for lyr in self.layers]))
        attn2 = list(itertools.chain.from_iterable([lyr.block2.attn.parameters() for lyr in self.layers]))
        ff1 = list(itertools.chain.from_iterable([lyr.block1.ff.parameters() for lyr in self.layers]))
        ff2 = list(itertools.chain.from_iterable([lyr.block2.ff.parameters() for lyr in self.layers]))
        blkout_layers = list(itertools.chain.from_iterable([lyr.out.parameters() for lyr in self.layers]))
        groups = {
            'prenorms': list(itertools.chain.from_iterable([lyr.prenorm.parameters() for lyr in self.layers])),
            'blk1_attention_layers': attn1,
            'blk2_attention_layers': attn2,
            'attention_layers': attn1 + attn2,
            'blk1_ff_layers': ff1,
            'blk2_ff_layers': ff2,
            'ff_layers': ff1 + ff2,
            'block_out_layers': blkout_layers,
            'out': list(self.out.parameters()),
            'x_proj': list(self.inp_block.parameters()),
            'layers': list(self.layers.parameters()),
            'time_embed': list(self.time_embed.parameters()),
            'resolution_embed': list(self.resolution_embed.parameters()),
        }
        return groups

    def input_to_random_resolution_and_window(self, x):
        """
        This function MUST be applied to the target *before* noising. It returns the reduced, re-scoped target as well
        as caches an internal prior for the rescoped target which will be used in training.
        Args:
            x: Diffusion target
        """
        resolution = randrange(0, self.resolution_steps)
        resolution_scale = 2 ** resolution
        s = F.interpolate(x, scale_factor=1/resolution_scale, mode='nearest')
        s_diff = s.shape[-1] - self.max_window
        if s_diff > 1:
            start = randrange(0, s_diff)
            s = s[:,:,start:start+self.max_window]
        s_prior = F.interpolate(s, scale_factor=.25, mode='nearest')
        s_prior = F.interpolate(s_prior, size=(s.shape[-1],), mode='linear', align_corners=True)
        self.preprocessed = (s_prior, torch.tensor([resolution] * x.shape[0], dtype=torch.long, device=x.device))
        return s

    def forward(self, x, timesteps, x_prior=None, resolution=None, conditioning_input=None, conditioning_free=False):
        conditioning_input = x_prior if conditioning_input is None else conditioning_input

        h = x
        if resolution is None:
            # This is assumed to be training.
            assert self.preprocessed is not None, 'Preprocessing function not called.'
            assert x_prior is None, 'Provided prior will not be used, instead preprocessing output will be used.'
            h_sub, resolution = self.preprocessed
            self.preprocessed = None
        else:
            assert h.shape[-1] > x_prior.shape[-1] * 3.9, f'{h.shape} {x_prior.shape}'
            h_sub = F.interpolate(x_prior, size=(x.shape[-1],), mode='linear', align_corners=True)

        if conditioning_free:
            h_sub = self.unconditioned_prior.repeat(x.shape[0], 1, x.shape[-1])
            code_emb = self.unconditioned_embedding.repeat(x.shape[0], 1, x.shape[-1])
        else:
            MIN_COND_LEN = 200
            MAX_COND_LEN = 1200
            if self.training and conditioning_input.shape[-1] > MAX_COND_LEN:
                clen = randrange(MIN_COND_LEN, MAX_COND_LEN)
                gap = conditioning_input.shape[-1] - clen
                cstart = randrange(0, gap)
                conditioning_input = conditioning_input[:,:,cstart:cstart+clen]
            code_emb = self.conditioning_encoder(conditioning_input, resolution)

        # Mask out the conditioning input and x_prior inputs for whole batch elements, implementing something similar to classifier-free guidance.
        if self.training and self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand((h.shape[0], 1, 1),
                                               device=h.device) < self.unconditioned_percentage
            h_sub = torch.where(unconditioned_batches, self.unconditioned_prior.repeat(h_sub.shape[0], 1, h_sub.shape[-1]), h_sub)
            code_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(code_emb.shape[0], 1, 1), code_emb)

        with torch.autocast(x.device.type, enabled=self.enable_fp16):
            time_emb = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))
            res_emb = self.resolution_embed(resolution)
            blk_emb = torch.cat([time_emb.unsqueeze(-1), res_emb.unsqueeze(-1), code_emb], dim=-1)

            h = torch.cat([h, h_sub], dim=1)
            h = self.inp_block(h)
            for layer in self.layers:
                h = checkpoint(layer, h, blk_emb)

        h = h.float()
        out = self.out(h)

        # Defensively involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        unused_params = [self.unconditioned_prior, self.unconditioned_embedding]
        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        out = out + extraneous_addition * 0

        return out


@register_model
def register_transformer_diffusion13(opt_net, opt):
    return TransformerDiffusion(**opt_net['kwargs'])


def test_tfd():
    clip = torch.randn(2,256,10336)
    cond = torch.randn(2,256,10336)
    ts = torch.LongTensor([600, 600])
    model = TransformerDiffusion(in_channels=256, model_channels=1024, contraction_dim=512,
                                 num_heads=512//64, input_vec_dim=256, num_layers=12, dropout=.1,
                                 unconditioned_percentage=.6)
    for k in range(100):
        x = model.input_to_random_resolution_and_window(clip, x_prior=clip)
        model(x, ts, clip)


def remove_conditioning(sd_path):
    sd = torch.load(sd_path)
    del sd['unconditioned_embedding']
    torch.save(sd, sd_path.replace('.pth', '') + '_fixed.pth')


if __name__ == '__main__':
    remove_conditioning('X:\\dlas\\experiments\\train_music_diffusion_multilevel_sr_pre\\models\\12500_generator.pth')
    test_tfd()
