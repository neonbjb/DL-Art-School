import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio.music.music_quantizer2 import MusicQuantizer2
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
        self.ff = FeedForward(inp_dim+contraction_dim, contraction_dim, mult=1, dropout=dropout)

    def forward(self, x, rotary_emb):
        ah, _, _, _ = checkpoint(self.attn, x, None, None, None, None, None, rotary_emb)
        ah = F.gelu(self.attnorm(ah))
        h = torch.cat([ah, x], dim=-1)
        hf = checkpoint(self.ff, h)
        h = torch.cat([h, hf], dim=-1)
        return h

class DietAttentionBlock(TimestepBlock):
    def __init__(self, trunk_dim, heads, dropout):
        super().__init__()
        contraction_dim = trunk_dim // 4
        self.prenorm = RMSScaleShiftNorm(trunk_dim, bias=False)
        self.block1 = SubBlock(trunk_dim, contraction_dim, heads, dropout)
        self.block2 = SubBlock(trunk_dim+contraction_dim*2, contraction_dim, heads, dropout)
        self.out = nn.Linear(trunk_dim+contraction_dim*4, trunk_dim, bias=False)
        self.out.weight.data.zero_()

    def forward(self, x, timestep_emb, rotary_emb):
        h = self.prenorm(x, norm_scale_shift_inp=timestep_emb)
        h = self.block1(h, rotary_emb)
        h = self.block2(h, rotary_emb)
        h = self.out(h)
        return h + x


class TransformerDiffusion(nn.Module):
    """
    A diffusion model composed entirely of stacks of transformer layers. Why would you do it any other way?
    """
    def __init__(
            self,
            prenet_channels=256,
            prenet_layers=3,
            model_channels=512,
            num_layers=8,
            in_channels=256,
            rotary_emb_dim=32,
            input_vec_dim=512,
            out_channels=512,  # mean and variance
            num_heads=16,
            dropout=0,
            use_fp16=False,
            ar_prior=False,
            # Parameters for regularization.
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.prenet_channels = prenet_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16

        self.inp_block = conv_nd(1, in_channels, prenet_channels, 3, 1, 1)

        self.time_embed = nn.Sequential(
            linear(prenet_channels, prenet_channels),
            nn.SiLU(),
            linear(prenet_channels, model_channels),
        )

        self.ar_prior = ar_prior
        prenet_heads = prenet_channels//64
        if ar_prior:
            self.ar_input = nn.Linear(input_vec_dim, prenet_channels)
            self.ar_prior_intg = Encoder(
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
        else:
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
        self.layers = TimestepRotaryEmbedSequential(*[DietAttentionBlock(model_channels, num_heads, dropout) for _ in range(num_layers)])

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(conv_nd(1, model_channels, out_channels, 3, padding=1)),
        )

        self.debug_codes = {}

    def get_grad_norm_parameter_groups(self):
        groups = {
            'layers': list(self.layers.parameters()) + list(self.inp_block.parameters()),
            'code_converters': list(self.input_converter.parameters()) + list(self.code_converter.parameters()),
            'time_embed': list(self.time_embed.parameters()),
        }
        return groups

    def timestep_independent(self, prior, expected_seq_len):
        code_emb = self.ar_input(prior) if self.ar_prior else self.input_converter(prior)
        code_emb = self.ar_prior_intg(code_emb) if self.ar_prior else self.code_converter(code_emb)

        # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
        if self.training and self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand((code_emb.shape[0], 1, 1),
                                               device=code_emb.device) < self.unconditioned_percentage
            code_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(prior.shape[0], 1, 1),
                                   code_emb)

        expanded_code_emb = F.interpolate(code_emb.permute(0,2,1), size=expected_seq_len, mode='nearest').permute(0,2,1)
        return expanded_code_emb

    def forward(self, x, timesteps, codes=None, conditioning_input=None, precomputed_code_embeddings=None, conditioning_free=False):
        if precomputed_code_embeddings is not None:
            assert codes is None and conditioning_input is None, "Do not provide precomputed embeddings and the other parameters. It is unclear what you want me to do here."

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
            blk_emb = self.time_embed(timestep_embedding(timesteps, self.prenet_channels))
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
    def __init__(self, quantizer_dims=[1024], freeze_quantizer_until=20000, **kwargs):
        super().__init__()

        self.internal_step = 0
        self.freeze_quantizer_until = freeze_quantizer_until
        self.diff = TransformerDiffusion(**kwargs)
        self.quantizer = MusicQuantizer2(inp_channels=kwargs['in_channels'], inner_dim=quantizer_dims,
                                         codevector_dim=quantizer_dims[0], codebook_size=256,
                                         codebook_groups=2, max_gumbel_temperature=4, min_gumbel_temperature=.5)
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
        groups = {
            'blk1_attention_layers': attn1,
            'blk2_attention_layers': attn2,
            'attention_layers': attn1 + attn2,
            'blk1_ff_layers': ff1,
            'blk2_ff_layers': ff2,
            'ff_layers': ff1 + ff2,
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


class TransformerDiffusionWithARPrior(nn.Module):
    def __init__(self, freeze_diff=False, **kwargs):
        super().__init__()

        self.internal_step = 0
        from models.audio.music.gpt_music import GptMusicLower
        self.ar = GptMusicLower(dim=512, layers=12)
        for p in self.ar.parameters():
            p.DO_NOT_TRAIN = True
            p.requires_grad = False

        self.diff = TransformerDiffusion(ar_prior=True, **kwargs)
        if freeze_diff:
            for p in self.diff.parameters():
                p.DO_NOT_TRAIN = True
                p.requires_grad = False
            for p in list(self.diff.ar_prior_intg.parameters()) + list(self.diff.ar_input.parameters()):
                del p.DO_NOT_TRAIN
                p.requires_grad = True

    def get_grad_norm_parameter_groups(self):
        groups = {
            'attention_layers': list(itertools.chain.from_iterable([lyr.attn.parameters() for lyr in self.diff.layers])),
            'ff_layers': list(itertools.chain.from_iterable([lyr.ff.parameters() for lyr in self.diff.layers])),
            'rotary_embeddings': list(self.diff.rotary_embeddings.parameters()),
            'out': list(self.diff.out.parameters()),
            'x_proj': list(self.diff.inp_block.parameters()),
            'layers': list(self.diff.layers.parameters()),
            'ar_prior_intg': list(self.diff.ar_prior_intg.parameters()),
            'time_embed': list(self.diff.time_embed.parameters()),
        }
        return groups

    def forward(self, x, timesteps, truth_mel, disable_diversity=False, conditioning_input=None, conditioning_free=False):
        with torch.no_grad():
            prior = self.ar(truth_mel, conditioning_input, return_latent=True)

        diff = self.diff(x, timesteps, prior, conditioning_free=conditioning_free)
        return diff


@register_model
def register_transformer_diffusion9(opt_net, opt):
    return TransformerDiffusion(**opt_net['kwargs'])


@register_model
def register_transformer_diffusion10_with_quantizer(opt_net, opt):
    return TransformerDiffusionWithQuantizer(**opt_net['kwargs'])


@register_model
def register_transformer_diffusion10_with_ar_prior(opt_net, opt):
    return TransformerDiffusionWithARPrior(**opt_net['kwargs'])


def test_quant_model():
    clip = torch.randn(2, 256, 400)
    cond = torch.randn(2, 256, 400)
    ts = torch.LongTensor([600, 600])
    model = TransformerDiffusionWithQuantizer(in_channels=256, model_channels=1024,
                                              prenet_channels=1024, num_heads=8,
                                              input_vec_dim=1024, num_layers=20, prenet_layers=6,
                                              dropout=.1)

    quant_weights = torch.load('D:\\dlas\\experiments\\train_music_quant_r4\\models\\5000_generator.pth')
    model.quantizer.load_state_dict(quant_weights, strict=False)

    torch.save(model.state_dict(), 'sample.pth')
    print_network(model)
    o = model(clip, ts, clip, cond)
    model.get_grad_norm_parameter_groups()


def test_ar_model():
    clip = torch.randn(2, 256, 400)
    cond = torch.randn(2, 256, 400)
    ts = torch.LongTensor([600, 600])
    model = TransformerDiffusionWithARPrior(model_channels=2048, prenet_channels=1536,
                                            input_vec_dim=512, num_layers=16, prenet_layers=6, freeze_diff=True,
                                            unconditioned_percentage=.4)
    model.get_grad_norm_parameter_groups()

    ar_weights = torch.load('D:\\dlas\\experiments\\train_music_gpt\\models\\44500_generator_ema.pth')
    model.ar.load_state_dict(ar_weights, strict=True)
    diff_weights = torch.load('X:\\dlas\\experiments\\train_music_diffusion_tfd8\\models\\47500_generator_ema.pth')
    pruned_diff_weights = {}
    for k,v in diff_weights.items():
        if k.startswith('diff.'):
            pruned_diff_weights[k.replace('diff.', '')] = v
    model.diff.load_state_dict(pruned_diff_weights, strict=False)
    torch.save(model.state_dict(), 'sample.pth')

    model(clip, ts, cond, conditioning_input=cond)



if __name__ == '__main__':
    test_quant_model()
