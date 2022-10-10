import itertools
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.arch_util import ResBlock
from models.audio.music.gpt_music2 import UpperEncoder, GptMusicLower
from models.audio.music.music_quantizer2 import MusicQuantizer2
from models.audio.tts.lucidrains_dvae import DiscreteVAE
from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import TimestepBlock
from models.lucidrains.x_transformers import Encoder, Attention, RMSScaleShiftNorm, RotaryEmbedding, \
    FeedForward
from trainer.networks import register_model
from utils.util import checkpoint, print_network


def is_latent(t):
    return t.dtype == torch.float

def is_sequence(t):
    return t.dtype == torch.long


class MultiGroupEmbedding(nn.Module):
    def __init__(self, tokens, groups, dim):
        super().__init__()
        self.m = nn.ModuleList([nn.Embedding(tokens, dim // groups) for _ in range(groups)])

    def forward(self, x):
        h = [embedding(x[:, :, i]) for i, embedding in enumerate(self.m)]
        return torch.cat(h, dim=-1)


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
        self.ff = nn.Conv1d(inp_dim+contraction_dim, contraction_dim, kernel_size=3, padding=1)
        self.ffnorm = nn.LayerNorm(contraction_dim)

    def forward(self, x, rotary_emb):
        ah, _, _, _ = checkpoint(self.attn, x, None, None, None, None, None, rotary_emb)
        ah = F.gelu(self.attnorm(ah))
        h = torch.cat([ah, x], dim=-1)
        hf = checkpoint(self.ff, h.permute(0,2,1)).permute(0,2,1)
        hf = F.gelu(self.ffnorm(hf))
        h = torch.cat([h, hf], dim=-1)
        return h


class ConcatAttentionBlock(TimestepBlock):
    def __init__(self, trunk_dim, contraction_dim, time_embed_dim, heads, dropout):
        super().__init__()
        self.prenorm = RMSScaleShiftNorm(trunk_dim, embed_dim=time_embed_dim, bias=False)
        self.block1 = SubBlock(trunk_dim, contraction_dim, heads, dropout)
        self.block2 = SubBlock(trunk_dim+contraction_dim*2, contraction_dim, heads, dropout)
        self.out = nn.Linear(contraction_dim*4, trunk_dim, bias=False)
        self.out.weight.data.zero_()

    def forward(self, x, timestep_emb, rotary_emb):
        h = self.prenorm(x, norm_scale_shift_inp=timestep_emb)
        h = self.block1(h, rotary_emb)
        h = self.block2(h, rotary_emb)
        h = self.out(h[:,:,x.shape[-1]:])
        return h + x


class TransformerDiffusion(nn.Module):
    """
    A diffusion model composed entirely of stacks of transformer layers. Why would you do it any other way?
    """
    def __init__(
            self,
            prenet_channels=1024,
            prenet_layers=3,
            time_embed_dim=256,
            model_channels=1024,
            contraction_dim=256,
            num_layers=8,
            in_channels=256,
            rotary_emb_dim=32,
            input_vec_dim=1024,
            out_channels=512,  # mean and variance
            num_heads=4,
            dropout=0,
            use_corner_alignment=False,  # This is an interpolation parameter only provided for backwards compatibility. ALL NEW TRAINS SHOULD SET THIS TO TRUE.
            use_fp16=False,
            new_code_expansion=False,
            permute_codes=False,
            # Parameters for regularization.
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
            # Parameters for re-training head
            freeze_except_code_converters=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.prenet_channels = prenet_channels
        self.time_embed_dim = time_embed_dim
        self.out_channels = out_channels
        self.dropout = dropout
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.new_code_expansion = new_code_expansion
        self.permute_codes = permute_codes
        self.use_corner_alignment = use_corner_alignment

        self.inp_block = conv_nd(1, in_channels, prenet_channels, 3, 1, 1)

        self.time_embed = nn.Sequential(
            linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        prenet_heads = prenet_channels//64
        self.input_converter = nn.Linear(input_vec_dim, prenet_channels)
        self.code_converter = Encoder(
                    dim=prenet_channels,
                    depth=prenet_layers,
                    heads=prenet_heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                    zero_init_branch_output=True,
                    ff_mult=1,
                )

        self.unconditioned_embedding = nn.Parameter(torch.randn(1,1,prenet_channels))
        self.rotary_embeddings = RotaryEmbedding(rotary_emb_dim)
        self.intg = nn.Linear(prenet_channels*2, model_channels)
        self.layers = TimestepRotaryEmbedSequential(*[ConcatAttentionBlock(model_channels, contraction_dim, time_embed_dim, num_heads, dropout) for _ in range(num_layers)])

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(conv_nd(1, model_channels, out_channels, 3, padding=1)),
        )

        if freeze_except_code_converters:
            for p in self.parameters():
                p.DO_NOT_TRAIN = True
                p.requires_grad = False
            for m in [self.code_converter and self.input_converter]:
                for p in m.parameters():
                    del p.DO_NOT_TRAIN
                    p.requires_grad = True

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
            #'code_converters': list(self.input_converter.parameters()) + list(self.code_converter.parameters()),
            'time_embed': list(self.time_embed.parameters()),
        }
        return groups

    def timestep_independent(self, prior, expected_seq_len):
        if self.new_code_expansion:
            prior = F.interpolate(prior.permute(0,2,1), size=expected_seq_len, mode='linear', align_corners=self.use_corner_alignment).permute(0,2,1)
        code_emb = self.input_converter(prior)
        code_emb = self.code_converter(code_emb)

        # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
        if self.training and self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand((code_emb.shape[0], 1, 1),
                                               device=code_emb.device) < self.unconditioned_percentage
            code_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(prior.shape[0], 1, 1),
                                   code_emb)

        if not self.new_code_expansion:
            code_emb = F.interpolate(code_emb.permute(0,2,1), size=expected_seq_len, mode='nearest').permute(0,2,1)
        return code_emb

    def forward(self, x, timesteps, codes=None, conditioning_input=None, precomputed_code_embeddings=None, conditioning_free=False):
        if precomputed_code_embeddings is not None:
            assert codes is None and conditioning_input is None, "Do not provide precomputed embeddings and the other parameters. It is unclear what you want me to do here."
        if self.permute_codes:
            codes = codes.permute(0,2,1)

        unused_params = []
        if conditioning_free:
            code_emb = self.unconditioned_embedding.repeat(x.shape[0], x.shape[-1], 1)
        else:
            if precomputed_code_embeddings is not None:
                code_emb = precomputed_code_embeddings
            else:
                code_emb = self.timestep_independent(codes, x.shape[-1])
            unused_params.append(self.unconditioned_embedding)

        with torch.autocast(x.device.type, enabled=self.enable_fp16):
            blk_emb = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))
            x = self.inp_block(x).permute(0,2,1)

            rotary_pos_emb = self.rotary_embeddings(x.shape[1], x.device)
            x = self.intg(torch.cat([x, code_emb], dim=-1))
            for layer in self.layers:
                x = checkpoint(layer, x, blk_emb, rotary_pos_emb)

        x = x.float().permute(0,2,1)
        out = self.out(x)

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        out = out + extraneous_addition * 0

        return out


class TransformerDiffusionWithQuantizer(nn.Module):
    def __init__(self, quantizer_dims=[1024], quantizer_codebook_size=256, quantizer_codebook_groups=2,
                 freeze_quantizer_until=20000, **kwargs):
        super().__init__()

        self.internal_step = 0
        self.freeze_quantizer_until = freeze_quantizer_until
        self.diff = TransformerDiffusion(**kwargs)
        self.quantizer = MusicQuantizer2(inp_channels=kwargs['in_channels'], inner_dim=quantizer_dims,
                                         codevector_dim=quantizer_dims[0], codebook_size=quantizer_codebook_size,
                                         codebook_groups=quantizer_codebook_groups, max_gumbel_temperature=4,
                                         min_gumbel_temperature=.5)
        self.quantizer.quantizer.temperature = self.quantizer.min_gumbel_temperature
        del self.quantizer.up

    def update_for_step(self, step, *args):
        self.internal_step = step
        qstep = max(0, self.internal_step - self.freeze_quantizer_until)
        self.quantizer.quantizer.temperature = max(
            self.quantizer.max_gumbel_temperature * self.quantizer.gumbel_temperature_decay ** qstep,
                    self.quantizer.min_gumbel_temperature,
                )

    def forward(self, x, timesteps, truth_mel, conditioning_input=None, disable_diversity=False, conditioning_free=False):
        quant_grad_enabled = self.internal_step > self.freeze_quantizer_until
        with torch.set_grad_enabled(quant_grad_enabled):
            proj, diversity_loss = self.quantizer(truth_mel, return_decoder_latent=True)
            proj = proj.permute(0,2,1)

        # Make sure this does not cause issues in DDP by explicitly using the parameters for nothing.
        if not quant_grad_enabled:
            unused = 0
            for p in self.quantizer.parameters():
                unused = unused + p.mean() * 0
            proj = proj + unused
            diversity_loss = diversity_loss * 0

        diff = self.diff(x, timesteps, codes=proj, conditioning_input=conditioning_input, conditioning_free=conditioning_free)
        if disable_diversity:
            return diff
        return diff, diversity_loss

    def get_debug_values(self, step, __):
        if self.quantizer.total_codes > 0:
            return {'histogram_quant_codes': self.quantizer.codes[:self.quantizer.total_codes],
                    'gumbel_temperature': self.quantizer.quantizer.temperature}
        else:
            return {}

    def get_grad_norm_parameter_groups(self):
        attn1 = list(itertools.chain.from_iterable([lyr.block1.attn.parameters() for lyr in self.diff.layers]))
        attn2 = list(itertools.chain.from_iterable([lyr.block2.attn.parameters() for lyr in self.diff.layers]))
        ff1 = list(itertools.chain.from_iterable([lyr.block1.ff.parameters() for lyr in self.diff.layers]))
        ff2 = list(itertools.chain.from_iterable([lyr.block2.ff.parameters() for lyr in self.diff.layers]))
        blkout_layers = list(itertools.chain.from_iterable([lyr.out.parameters() for lyr in self.diff.layers]))
        groups = {
            'prenorms': list(itertools.chain.from_iterable([lyr.prenorm.parameters() for lyr in self.diff.layers])),
            'blk1_attention_layers': attn1,
            'blk2_attention_layers': attn2,
            'attention_layers': attn1 + attn2,
            'blk1_ff_layers': ff1,
            'blk2_ff_layers': ff2,
            'ff_layers': ff1 + ff2,
            'block_out_layers': blkout_layers,
            'quantizer_encoder': list(self.quantizer.encoder.parameters()),
            'quant_codebook': [self.quantizer.quantizer.codevectors],
            'rotary_embeddings': list(self.diff.rotary_embeddings.parameters()),
            'out': list(self.diff.out.parameters()),
            'x_proj': list(self.diff.inp_block.parameters()),
            'layers': list(self.diff.layers.parameters()),
            'code_converters': list(self.diff.input_converter.parameters()) + list(self.diff.code_converter.parameters()),
            'time_embed': list(self.diff.time_embed.parameters()),
        }
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


class TransformerDiffusionWithPretrainedVqvae(nn.Module):
    def __init__(self, vqargs, **kwargs):
        super().__init__()

        self.internal_step = 0
        self.diff = TransformerDiffusion(**kwargs)
        self.quantizer = DiscreteVAE(**vqargs)
        self.quantizer = self.quantizer.eval()
        for p in self.quantizer.parameters():
            p.DO_NOT_TRAIN = True
            p.requires_grad = False

    def forward(self, x, timesteps, truth_mel, conditioning_input=None, disable_diversity=False, conditioning_free=False):
        with torch.no_grad():
            reconstructed, proj = self.quantizer.infer(truth_mel)
            proj = proj.permute(0,2,1)

        diff = self.diff(x, timesteps, codes=proj, conditioning_input=conditioning_input, conditioning_free=conditioning_free)
        return diff

    def get_debug_values(self, step, __):
        if self.quantizer.total_codes > 0:
            return {'histogram_quant_codes': self.quantizer.codes[:self.quantizer.total_codes]}
        else:
            return {}

    def get_grad_norm_parameter_groups(self):
        attn1 = list(itertools.chain.from_iterable([lyr.block1.attn.parameters() for lyr in self.diff.layers]))
        attn2 = list(itertools.chain.from_iterable([lyr.block2.attn.parameters() for lyr in self.diff.layers]))
        ff1 = list(itertools.chain.from_iterable([lyr.block1.ff.parameters() for lyr in self.diff.layers]))
        ff2 = list(itertools.chain.from_iterable([lyr.block2.ff.parameters() for lyr in self.diff.layers]))
        blkout_layers = list(itertools.chain.from_iterable([lyr.out.parameters() for lyr in self.diff.layers]))
        groups = {
            'prenorms': list(itertools.chain.from_iterable([lyr.prenorm.parameters() for lyr in self.diff.layers])),
            'blk1_attention_layers': attn1,
            'blk2_attention_layers': attn2,
            'attention_layers': attn1 + attn2,
            'blk1_ff_layers': ff1,
            'blk2_ff_layers': ff2,
            'ff_layers': ff1 + ff2,
            'block_out_layers': blkout_layers,
            'rotary_embeddings': list(self.diff.rotary_embeddings.parameters()),
            'out': list(self.diff.out.parameters()),
            'x_proj': list(self.diff.inp_block.parameters()),
            'layers': list(self.diff.layers.parameters()),
            #'code_converters': list(self.diff.input_converter.parameters()) + list(self.diff.code_converter.parameters()),
            'time_embed': list(self.diff.time_embed.parameters()),
        }
        return groups

    def before_step(self, step):
        scaled_grad_parameters = list(itertools.chain.from_iterable([lyr.out.parameters() for lyr in self.diff.layers])) + \
                                 list(itertools.chain.from_iterable([lyr.prenorm.parameters() for lyr in self.diff.layers]))
        # Scale back the gradients of the blkout and prenorm layers by a constant factor. These get two orders of magnitudes
        # higher gradients. Ideally we would use parameter groups, but ZeroRedundancyOptimizer makes this trickier than
        # directly fiddling with the gradients.
        for p in scaled_grad_parameters:
            p.grad *= .2


class TransformerDiffusionWithMultiPretrainedVqvae(nn.Module):
    def __init__(self, num_vaes=4, vqargs={}, **kwargs):
        super().__init__()

        self.internal_step = 0
        self.diff = TransformerDiffusion(**kwargs)
        self.quantizers = nn.ModuleList([DiscreteVAE(**vqargs).eval() for _ in range(num_vaes)])
        for p in self.quantizers.parameters():
            p.DO_NOT_TRAIN = True
            p.requires_grad = False

    def forward(self, x, timesteps, truth_mel, conditioning_input=None, disable_diversity=False, conditioning_free=False):
        with torch.no_grad():
            proj = []
            partition_size = truth_mel.shape[1] // len(self.quantizers)
            for i, q in enumerate(self.quantizers):
                mel_partition = truth_mel[:, i*partition_size:(i+1)*partition_size]
                _, p = q.infer(mel_partition)
                proj.append(p.permute(0,2,1))
            proj = torch.cat(proj, dim=-1)

        diff = self.diff(x, timesteps, codes=proj, conditioning_input=conditioning_input, conditioning_free=conditioning_free)
        return diff

    def get_debug_values(self, step, __):
        if self.quantizers[0].total_codes > 0:
            dbgs = {}
            for i in range(len(self.quantizers)):
                dbgs[f'histogram_quant{i}_codes'] = self.quantizers[i].codes[:self.quantizers[i].total_codes]
            return dbgs
        else:
            return {}

    def get_grad_norm_parameter_groups(self):
        attn1 = list(itertools.chain.from_iterable([lyr.block1.attn.parameters() for lyr in self.diff.layers]))
        attn2 = list(itertools.chain.from_iterable([lyr.block2.attn.parameters() for lyr in self.diff.layers]))
        ff1 = list(itertools.chain.from_iterable([lyr.block1.ff.parameters() for lyr in self.diff.layers]))
        ff2 = list(itertools.chain.from_iterable([lyr.block2.ff.parameters() for lyr in self.diff.layers]))
        blkout_layers = list(itertools.chain.from_iterable([lyr.out.parameters() for lyr in self.diff.layers]))
        groups = {
            'prenorms': list(itertools.chain.from_iterable([lyr.prenorm.parameters() for lyr in self.diff.layers])),
            'blk1_attention_layers': attn1,
            'blk2_attention_layers': attn2,
            'attention_layers': attn1 + attn2,
            'blk1_ff_layers': ff1,
            'blk2_ff_layers': ff2,
            'ff_layers': ff1 + ff2,
            'block_out_layers': blkout_layers,
            'rotary_embeddings': list(self.diff.rotary_embeddings.parameters()),
            'out': list(self.diff.out.parameters()),
            'x_proj': list(self.diff.inp_block.parameters()),
            'layers': list(self.diff.layers.parameters()),
            'code_converters': list(self.diff.input_converter.parameters()) + list(self.diff.code_converter.parameters()),
            'time_embed': list(self.diff.time_embed.parameters()),
        }
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

class TransformerDiffusionWithCheaterLatent(nn.Module):
    def __init__(self, freeze_encoder_until=None, checkpoint_encoder=True, res_encoder=False, **kwargs):
        super().__init__()
        self.internal_step = 0
        self.freeze_encoder_until = freeze_encoder_until
        self.diff = TransformerDiffusion(**kwargs)
        if res_encoder:
            from models.audio.music.encoders import ResEncoder16x
            self.encoder = ResEncoder16x(256, 1024, 256, checkpointing_enabled=checkpoint_encoder)
        else:
            self.encoder = UpperEncoder(256, 1024, 256).eval()

    def forward(self, x, timesteps, truth_mel, conditioning_input=None, disable_diversity=False, conditioning_free=False):
        unused_parameters = []
        encoder_grad_enabled = self.freeze_encoder_until is not None and self.internal_step > self.freeze_encoder_until
        if not encoder_grad_enabled:
            unused_parameters.extend(list(self.encoder.parameters()))
        with torch.set_grad_enabled(encoder_grad_enabled):
            proj = self.encoder(truth_mel).permute(0,2,1)

        for p in unused_parameters:
            proj = proj + p.mean() * 0

        diff = self.diff(x, timesteps, codes=proj, conditioning_input=conditioning_input, conditioning_free=conditioning_free)
        return diff

    def get_debug_values(self, step, __):
        self.internal_step = step
        return {}

    def get_grad_norm_parameter_groups(self):
        attn1 = list(itertools.chain.from_iterable([lyr.block1.attn.parameters() for lyr in self.diff.layers]))
        attn2 = list(itertools.chain.from_iterable([lyr.block2.attn.parameters() for lyr in self.diff.layers]))
        ff1 = list(itertools.chain.from_iterable([lyr.block1.ff.parameters() for lyr in self.diff.layers]))
        ff2 = list(itertools.chain.from_iterable([lyr.block2.ff.parameters() for lyr in self.diff.layers]))
        blkout_layers = list(itertools.chain.from_iterable([lyr.out.parameters() for lyr in self.diff.layers]))
        groups = {
            'prenorms': list(itertools.chain.from_iterable([lyr.prenorm.parameters() for lyr in self.diff.layers])),
            'blk1_attention_layers': attn1,
            'blk2_attention_layers': attn2,
            'attention_layers': attn1 + attn2,
            'blk1_ff_layers': ff1,
            'blk2_ff_layers': ff2,
            'ff_layers': ff1 + ff2,
            'block_out_layers': blkout_layers,
            'rotary_embeddings': list(self.diff.rotary_embeddings.parameters()),
            'out': list(self.diff.out.parameters()),
            'x_proj': list(self.diff.inp_block.parameters()),
            'layers': list(self.diff.layers.parameters()),
            'code_converters': list(self.diff.input_converter.parameters()) + list(self.diff.code_converter.parameters()),
            'time_embed': list(self.diff.time_embed.parameters()),
            'encoder': list(self.encoder.parameters()),
        }
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
def register_transformer_diffusion12(opt_net, opt):
    return TransformerDiffusion(**opt_net['kwargs'])


@register_model
def register_transformer_diffusion12_with_quantizer(opt_net, opt):
    return TransformerDiffusionWithQuantizer(**opt_net['kwargs'])


@register_model
def register_transformer_diffusion_12_with_pretrained_vqvae(opt_net, opt):
    return TransformerDiffusionWithPretrainedVqvae(**opt_net['kwargs'])


@register_model
def register_transformer_diffusion_12_with_multi_vqvae(opt_net, opt):
    return TransformerDiffusionWithMultiPretrainedVqvae(**opt_net['kwargs'])


@register_model
def register_transformer_diffusion_12_with_cheater_latent(opt_net, opt):
    return TransformerDiffusionWithCheaterLatent(**opt_net['kwargs'])


def test_tfd():
    clip = torch.randn(2,256,400)
    ts = torch.LongTensor([600, 600])
    model = TransformerDiffusion(in_channels=256, model_channels=1024, contraction_dim=512,
                                              prenet_channels=1024, num_heads=3, permute_codes=True,
                                              input_vec_dim=256, num_layers=12, prenet_layers=4,
                                              dropout=.1)
    model(clip, ts, clip)

def test_quant_model():
    clip = torch.randn(2, 256, 400)
    ts = torch.LongTensor([600, 600])

    # For music:
    model = TransformerDiffusionWithQuantizer(in_channels=256, model_channels=1536, contraction_dim=768,
                                              prenet_channels=1024, num_heads=10,
                                              input_vec_dim=1024, num_layers=24, prenet_layers=4,
                                              dropout=.1)
    quant_weights = torch.load('D:\\dlas\\experiments\\train_music_quant_r4\\models\\5000_generator.pth')
    model.quantizer.load_state_dict(quant_weights, strict=False)
    torch.save(model.state_dict(), 'sample.pth')

    print_network(model)
    o = model(clip, ts, clip)
    pg = model.get_grad_norm_parameter_groups()
    t = 0
    for k, vs in pg.items():
        s = 0
        for v in vs:
            m = 1
            for d in v.shape:
                m *= d
            s += m
        t += s
        print(k, s/1000000)
    print(t)


def test_vqvae_model():
    clip = torch.randn(2, 100, 400)
    cond = torch.randn(2,80,400)
    ts = torch.LongTensor([600, 600])

    # For music:
    model = TransformerDiffusionWithPretrainedVqvae(in_channels=100, out_channels=200,
                                                    model_channels=1024, contraction_dim=512,
                                              prenet_channels=1024, num_heads=8,
                                              input_vec_dim=512, num_layers=12, prenet_layers=6,
                                              dropout=.1, vqargs= {
                                                     'positional_dims': 1, 'channels': 80,
            'hidden_dim': 512, 'num_resnet_blocks': 3, 'codebook_dim': 512, 'num_tokens': 8192,
            'num_layers': 2, 'record_codes': True, 'kernel_size': 3, 'use_transposed_convs': False,
                                                }
                                              )
    quant_weights = torch.load('D:\\dlas\\experiments\\retrained_dvae_8192_clips.pth')
    model.quantizer.load_state_dict(quant_weights, strict=True)
    torch.save(model.state_dict(), 'sample.pth')

    print_network(model)
    o = model(clip, ts, cond)
    pg = model.get_grad_norm_parameter_groups()

    with torch.no_grad():
        proj = torch.randn(2, 100, 512).cuda()
        clip = clip.cuda()
        ts = ts.cuda()
        start = time()
        model = model.cuda().eval()
        model.diff.enable_fp16 = True
        ti = model.diff.timestep_independent(proj, clip.shape[2])
        for k in range(1000):
            model.diff(clip, ts, precomputed_code_embeddings=ti)
        print(f"Elapsed: {time()-start}")


def test_multi_vqvae_model():
    clip = torch.randn(2, 256, 400)
    cond = torch.randn(2,256,400)
    ts = torch.LongTensor([600, 600])

    # For music:
    model = TransformerDiffusionWithMultiPretrainedVqvae(in_channels=256, out_channels=512,
                                                    model_channels=1024, contraction_dim=512,
                                              prenet_channels=1024, num_heads=8,
                                              input_vec_dim=2048, num_layers=12, prenet_layers=6,
                                              dropout=.1, vqargs= {
                                                     'positional_dims': 1, 'channels': 64,
            'hidden_dim': 512, 'num_resnet_blocks': 3, 'codebook_dim': 512, 'num_tokens': 8192,
            'num_layers': 0, 'record_codes': True, 'kernel_size': 3, 'use_transposed_convs': False,
                                                }, num_vaes=4,
                                              )
    quants = ['X:\\dlas\\experiments\\music_vqvaes\\train_lrdvae_music_low\\models\\7500_generator.pth',
              'X:\\dlas\\experiments\\music_vqvaes\\train_lrdvae_music_mid_low\\models\\11000_generator.pth',
              'X:\\dlas\\experiments\\music_vqvaes\\train_lrdvae_music_mid_high\\models\\11500_generator.pth',
              'X:\\dlas\\experiments\\music_vqvaes\\train_lrdvae_music_high\\models\\11500_generator.pth']
    for i, qfile in enumerate(quants):
        quant_weights = torch.load(qfile)
        model.quantizers[i].load_state_dict(quant_weights, strict=True)
    torch.save(model.state_dict(), 'sample.pth')

    print_network(model)
    o = model(clip, ts, cond)
    pg = model.get_grad_norm_parameter_groups()
    model.diff.get_grad_norm_parameter_groups()


def test_cheater_model():
    clip = torch.randn(2, 256, 400)
    ts = torch.LongTensor([600, 600])

    # For music:
    model = TransformerDiffusionWithCheaterLatent(in_channels=256, out_channels=512,
                                                    model_channels=1024, contraction_dim=512,
                                                    prenet_channels=1024, num_heads=8,
                                                    input_vec_dim=256, num_layers=16, prenet_layers=6,
                                                    dropout=.1, new_code_expansion=True,
                                              )
    #diff_weights = torch.load('extracted_diff.pth')
    #model.diff.load_state_dict(diff_weights, strict=False)
    model.encoder.load_state_dict(torch.load('../experiments/music_cheater_encoder_256.pth', map_location=torch.device('cpu')), strict=True)
    torch.save(model.state_dict(), 'sample.pth')

    print_network(model)
    o = model(clip, ts, clip)
    pg = model.get_grad_norm_parameter_groups()


def extract_diff(in_f, out_f, remove_head=False):
    p = torch.load(in_f)
    out = {}
    for k, v in p.items():
        if k.startswith('diff.'):
            if remove_head and (k.startswith('diff.input_converter') or k.startswith('diff.code_converter')):
                continue
            out[k.replace('diff.', '')] = v
    torch.save(out, out_f)


if __name__ == '__main__':
    #extract_diff('X:\\dlas\\experiments\\train_music_diffusion_tfd12\\models\\41000_generator_ema.pth', 'extracted_diff.pth', True)
    #test_cheater_model()
    test_vqvae_model()
    #extract_diff('X:\\dlas\experiments\\train_music_diffusion_tfd_cheater_from_scratch\\models\\56500_generator_ema.pth', 'extracted.pth', remove_head=True)
