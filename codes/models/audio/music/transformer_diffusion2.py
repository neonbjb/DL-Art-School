import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import TimestepEmbedSequential, TimestepBlock
from models.lucidrains.x_transformers import Encoder, Attention, FeedForward, RMSScaleShiftNorm, RotaryEmbedding
from trainer.networks import register_model
from utils.util import checkpoint


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


class AttentionBlock(TimestepBlock):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = Attention(dim, heads=heads, causal=False, dropout=dropout, zero_init_output=False)
        self.ff = FeedForward(dim, mult=2, dropout=dropout, zero_init_output=True)
        self.rms_scale_norm = RMSScaleShiftNorm(dim)

    def forward(self, x, timestep_emb, rotary_emb):
        h = self.rms_scale_norm(x, norm_scale_shift_inp=timestep_emb)
        h, _, _, _ = checkpoint(self.attn, h, None, None, None, None, None, rotary_emb)
        h = checkpoint(self.ff, h)
        return h + x


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
            rotary_emb_dim=32,
            token_count=8,
            in_groups=None,
            out_channels=512,  # mean and variance
            dropout=0,
            use_fp16=False,
            # Parameters for regularization.
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
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
        if in_groups is None:
            self.embeddings = nn.Embedding(token_count, model_channels)
        else:
            self.embeddings = MultiGroupEmbedding(token_count, in_groups, model_channels)
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
        self.mel_head = nn.Conv1d(model_channels, in_channels, kernel_size=3, padding=1)

        self.rotary_embeddings = RotaryEmbedding(rotary_emb_dim)
        self.top_layers = TimestepRotaryEmbedSequential(*[AttentionBlock(model_channels, model_channels//64, dropout) for _ in range(num_layers//4)])
        self.mid_intg = nn.Linear(model_channels*2, model_channels, bias=False)
        self.mid_layers = TimestepRotaryEmbedSequential(*[AttentionBlock(model_channels, model_channels//64, dropout) for _ in range(num_layers//2)])
        self.final_intg = nn.Linear(model_channels*2, model_channels, bias=False)
        self.final_layers = TimestepRotaryEmbedSequential(*[AttentionBlock(model_channels, model_channels//64, dropout) for _ in range(num_layers//4)])

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(conv_nd(1, model_channels, out_channels, 3, padding=1)),
        )

        self.debug_codes = {}

    def get_grad_norm_parameter_groups(self):
        groups = {
            'contextual_embedder': list(self.conditioning_embedder.parameters()),
            'top_layers': list(self.top_layers.parameters()) + list(self.inp_block.parameters()),
            'mid_layers': list(self.mid_layers.parameters()),
            'final_layers': list(self.final_layers.parameters()),
            'code_converters': list(self.embeddings.parameters()) + list(self.code_converter.parameters()) + list(self.latent_conditioner.parameters()),
            'time_embed': list(self.time_embed.parameters()),
        }
        return groups

    def timestep_independent(self, codes, conditioning_input, expected_seq_len, prenet_latent=None, return_code_pred=False):
        cond_emb = self.conditioning_embedder(conditioning_input).permute(0,2,1)
        cond_emb = self.conditioning_encoder(cond_emb)[:, 0]

        code_emb = self.embeddings(codes)
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


    def forward(self, x, timesteps, codes=None, conditioning_input=None, prenet_latent=None, precomputed_code_embeddings=None,
                precomputed_cond_embeddings=None, conditioning_free=False, return_code_pred=False):
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

        rotary_pos_emb = self.rotary_embeddings(x.shape[1], x.device)
        xt = self.top_layers(x, blk_emb, rotary_pos_emb)
        xm = torch.cat([xt, code_emb], dim=2)
        xm = self.mid_intg(xm)
        xm = self.mid_layers(xm, blk_emb, rotary_pos_emb)
        xb = torch.cat([xt, xm], dim=2)
        xb = self.final_intg(xb)
        x = self.final_layers(xb, blk_emb, rotary_pos_emb)

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


@register_model
def register_transformer_diffusion2(opt_net, opt):
    return TransformerDiffusion(**opt_net['kwargs'])


if __name__ == '__main__':
    clip = torch.randn(2, 256, 400)
    aligned_latent = torch.randn(2,100,512)
    aligned_sequence = torch.randint(0,8,(2,100,8))
    cond = torch.randn(2, 256, 400)
    ts = torch.LongTensor([600, 600])
    model = TransformerDiffusion(512, unconditioned_percentage=.5, in_groups=8)
    o = model(clip, ts, aligned_sequence, cond, return_code_pred=True)
    #o = model(clip, ts, aligned_sequence, cond, aligned_latent)

