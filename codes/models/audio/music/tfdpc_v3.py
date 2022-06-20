import itertools
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.arch_util import ResBlock, AttentionBlock
from models.audio.music.gpt_music2 import UpperEncoder, GptMusicLower
from models.audio.music.music_quantizer2 import MusicQuantizer2
from models.audio.tts.lucidrains_dvae import DiscreteVAE
from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import TimestepBlock
from models.lucidrains.x_transformers import Encoder, Attention, RMSScaleShiftNorm, RotaryEmbedding, \
    FeedForward
from trainer.networks import register_model
from utils.util import checkpoint, print_network


class TimestepRotaryEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, rotary_emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, rotary_emb)
            else:
                x = layer(x, rotary_emb)
        return x


class SubBlock(nn.Module):
    def __init__(self, inp_dim, contraction_dim, heads, dropout):
        super().__init__()
        self.attn = Attention(inp_dim, out_dim=contraction_dim, heads=heads, dim_head=contraction_dim//heads, causal=False, dropout=dropout)
        self.attnorm = nn.LayerNorm(contraction_dim)
        self.ff = FeedForward(inp_dim+contraction_dim, dim_out=contraction_dim, mult=2, dropout=dropout)
        self.ffnorm = nn.LayerNorm(contraction_dim)

    def forward(self, x, rotary_emb):
        ah, _, _, _ = checkpoint(self.attn, x, None, None, None, None, None, rotary_emb)
        ah = F.gelu(self.attnorm(ah))
        h = torch.cat([ah, x], dim=-1)
        hf = checkpoint(self.ff, h)
        hf = F.gelu(self.ffnorm(hf))
        h = torch.cat([h, hf], dim=-1)
        return h


class ConcatAttentionBlock(TimestepBlock):
    def __init__(self, trunk_dim, contraction_dim, time_embed_dim, cond_dim_in, cond_dim_hidden, heads, dropout):
        super().__init__()
        self.prenorm = RMSScaleShiftNorm(trunk_dim, embed_dim=time_embed_dim, bias=False)
        self.cond_project = nn.Linear(cond_dim_in, cond_dim_hidden)
        self.block1 = SubBlock(trunk_dim+cond_dim_hidden, contraction_dim, heads, dropout)
        self.block2 = SubBlock(trunk_dim+cond_dim_hidden+contraction_dim*2, contraction_dim, heads, dropout)
        self.out = nn.Linear(contraction_dim*4, trunk_dim, bias=False)
        self.out.weight.data.zero_()

    def forward(self, x, cond, timestep_emb, rotary_emb):
        h = self.prenorm(x, norm_scale_shift_inp=timestep_emb)
        cond = self.cond_project(cond)
        h = torch.cat([h, cond], dim=-1)
        h = self.block1(h, rotary_emb)
        h = self.block2(h, rotary_emb)
        h = self.out(h[:,:,x.shape[-1]+cond.shape[-1]:])
        return h + x


class TransformerDiffusionWithPointConditioning(nn.Module):
    """
    A diffusion model composed entirely of stacks of transformer layers. Why would you do it any other way?
    """
    def __init__(
            self,
            in_channels=256,
            out_channels=512,  # mean and variance
            model_channels=1024,
            contraction_dim=256,
            time_embed_dim=256,
            num_layers=8,
            rotary_emb_dim=32,
            input_cond_dim=1024,
            num_heads=8,
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

        self.inp_block = conv_nd(1, in_channels, model_channels, 3, 1, 1)

        self.time_embed = nn.Sequential(
            linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.unconditioned_embedding = nn.Parameter(torch.randn(1,1,model_channels))
        self.rotary_embeddings = RotaryEmbedding(rotary_emb_dim)
        self.layers = TimestepRotaryEmbedSequential(*[ConcatAttentionBlock(model_channels,
                                                                           contraction_dim,
                                                                           time_embed_dim,
                                                                           cond_dim_in=input_cond_dim,
                                                                           cond_dim_hidden=input_cond_dim//2,
                                                                           heads=num_heads,
                                                                           dropout=dropout) for _ in range(num_layers)])

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
            'rotary_embeddings': list(self.rotary_embeddings.parameters()),
            'out': list(self.out.parameters()),
            'x_proj': list(self.inp_block.parameters()),
            'layers': list(self.layers.parameters()),
            'time_embed': list(self.time_embed.parameters()),
        }
        return groups

    def forward(self, x, timesteps, conditioning_input, conditioning_free=False):
        unused_params = []
        if conditioning_free:
            cond = self.unconditioned_embedding.repeat(x.shape[0], x.shape[-1], 1)
        else:
            cond = conditioning_input
            # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
            if self.training and self.unconditioned_percentage > 0:
                unconditioned_batches = torch.rand((cond.shape[0], 1, 1),
                                                   device=cond.device) < self.unconditioned_percentage
                cond = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(cond.shape[0], 1, 1), cond)
            unused_params.append(self.unconditioned_embedding)
        cond = cond.repeat(1,x.shape[-1],1)

        with torch.autocast(x.device.type, enabled=self.enable_fp16):
            blk_emb = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))
            x = self.inp_block(x).permute(0,2,1)

            rotary_pos_emb = self.rotary_embeddings(x.shape[1]+1, x.device)
            for layer in self.layers:
                x = checkpoint(layer, x, cond, blk_emb, rotary_pos_emb)

        x = x.float().permute(0,2,1)
        out = self.out(x)

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        out = out + extraneous_addition * 0

        return out


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 cond_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=8,
                 dropout=.1,
                 do_checkpointing=False):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(cond_dim, embedding_dim, kernel_size=1)
        self.attn = Encoder(
                dim=embedding_dim,
                depth=attn_blocks,
                heads=num_attn_heads,
                ff_dropout=dropout,
                attn_dropout=dropout,
                use_rmsnorm=True,
                ff_glu=True,
                rotary_pos_emb=True,
                zero_init_branch_output=True,
                ff_mult=2,
            )
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing

    def forward(self, x):
        h = self.init(x).permute(0,2,1)
        h = self.attn(h).permute(0,2,1)
        return h.mean(dim=2).unsqueeze(1)


class TransformerDiffusionWithConditioningEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.internal_step = 0
        self.diff = TransformerDiffusionWithPointConditioning(**kwargs)
        self.conditioning_encoder = ConditioningEncoder(256, kwargs['model_channels'])

    def forward(self, x, timesteps, true_cheater, conditioning_input=None, disable_diversity=False, conditioning_free=False):
        cond = self.conditioning_encoder(true_cheater)
        diff = self.diff(x, timesteps, conditioning_input=cond, conditioning_free=conditioning_free)
        return diff

    def get_debug_values(self, step, __):
        self.internal_step = step
        return {}

    def get_grad_norm_parameter_groups(self):
        groups = self.diff.get_grad_norm_parameter_groups()
        groups['conditioning_encoder'] = list(self.conditioning_encoder.parameters())
        return groups

    def before_step(self, step):
        scaled_grad_parameters = list(itertools.chain.from_iterable([lyr.out.parameters() for lyr in self.diff.layers])) + \
                                 list(itertools.chain.from_iterable([lyr.prenorm.parameters() for lyr in self.diff.layers]))
        # Scale back the gradients of the blkout and prenorm layers by a constant factor. These get two orders of magnitudes
        # higher gradients. Ideally we would use parameter groups, but ZeroRedundancyOptimizer makes this trickier than
        # directly fiddling with the gradients.
        for p in scaled_grad_parameters:
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad *= .2


@register_model
def register_tfdpc2(opt_net, opt):
    return TransformerDiffusionWithPointConditioning(**opt_net['kwargs'])


@register_model
def register_tfdpc3_with_conditioning_encoder(opt_net, opt):
    return TransformerDiffusionWithConditioningEncoder(**opt_net['kwargs'])


def test_cheater_model():
    clip = torch.randn(2, 256, 400)
    cl = torch.randn(2, 256, 400)
    ts = torch.LongTensor([600, 600])

    # For music:
    model = TransformerDiffusionWithConditioningEncoder(model_channels=1024)
    print_network(model)
    o = model(clip, ts, cl)
    pg = model.get_grad_norm_parameter_groups()


if __name__ == '__main__':
    test_cheater_model()
