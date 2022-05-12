from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum, distributed
from torch.distributed import get_world_size

from models.arch_util import AttentionBlock
from models.lucidrains.x_transformers import ContinuousTransformerWrapper, Encoder
from trainer.networks import register_model
from utils.util import opt_get, checkpoint


def exists(val):
    return val is not None


def masked_mean(t, mask):
    t = t.masked_fill(~mask, 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)


class CollapsingTransformer(nn.Module):
    def __init__(self, model_dim, output_dims, heads, dropout, depth, mask_percentage=0, **encoder_kwargs):
        super().__init__()
        self.transformer = ContinuousTransformerWrapper(
            max_seq_len=-1,
            use_pos_emb=False,
            attn_layers=Encoder(
                dim=model_dim,
                depth=depth,
                heads=heads,
                ff_dropout=dropout,
                ff_mult=1,
                attn_dropout=dropout,
                use_rmsnorm=True,
                ff_glu=True,
                rotary_pos_emb=True,
                **encoder_kwargs,
            ))
        self.pre_combiner = nn.Sequential(nn.Conv1d(model_dim, output_dims, 1),
                                          AttentionBlock(output_dims, num_heads=heads, do_checkpoint=False),
                                          nn.Conv1d(output_dims, output_dims, 1))
        self.mask_percentage = mask_percentage

    def forward(self, x, **transformer_kwargs):
        h = self.transformer(x, **transformer_kwargs)
        h = h.permute(0,2,1)
        h = checkpoint(self.pre_combiner, h).permute(0,2,1)
        if self.training:
            mask = torch.rand_like(h.float()) > self.mask_percentage
        else:
            mask = torch.ones_like(h.float()).bool()
        return masked_mean(h, mask)


class ConvFormatEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(*args, **kwargs)

    def forward(self, x):
        y = self.emb(x)
        return y.permute(0,2,1)


class CLVP(nn.Module):
    """
    Contrastic Language-Voice Pretraining model for generating embedding that can be used to associate text and
    speech clips.
    """

    def __init__(
            self,
            model_dim=512,
            transformer_heads=8,
            dropout=.1,
            num_text_tokens=256,
            text_enc_depth=6,
            text_mask_percentage=0,
            conditioning_enc_depth=4,
            mask_conditioning_percentage=0.5,
            mel_channels=80,
            mel_codes=None,
            speech_enc_depth=6,
            speech_mask_percentage=0,
            latent_multiplier=4,
            distributed_collect=False,
    ):
        super().__init__()
        latent_dim = latent_multiplier*model_dim
        self.temperature = nn.Parameter(torch.tensor(1.))

        self.cond_emb = nn.Sequential(nn.Conv1d(mel_channels, model_dim//2, kernel_size=5, stride=2, padding=2),
                                      nn.Conv1d(model_dim//2, model_dim, kernel_size=3, stride=2, padding=1))
        self.conditioning_transformer = CollapsingTransformer(model_dim, model_dim*2, transformer_heads, dropout, conditioning_enc_depth, 0)
        self.masked_conditioning_latent = nn.Parameter(torch.randn(1,model_dim*2), requires_grad=True)
        self.mask_conditioning_percentage = mask_conditioning_percentage

        self.text_emb = nn.Embedding(num_text_tokens, model_dim)
        self.text_transformer = CollapsingTransformer(model_dim, latent_dim, transformer_heads, dropout, text_enc_depth, text_mask_percentage, use_rms_scaleshift_norm=True)
        self.to_text_latent = nn.Linear(latent_dim, latent_dim, bias=False)
        self.distributed_collect = distributed_collect

        if mel_codes is None:
            self.speech_emb = nn.Conv1d(mel_channels, model_dim, kernel_size=5, padding=2)
        else:
            self.speech_emb = ConvFormatEmbedding(mel_codes, model_dim)
        self.speech_transformer = CollapsingTransformer(model_dim, latent_dim, transformer_heads, dropout, speech_enc_depth, speech_mask_percentage)
        self.to_speech_latent = nn.Linear(latent_dim, latent_dim, bias=False)

    def get_grad_norm_parameter_groups(self):
        return {
            'conditioning': list(self.conditioning_transformer.parameters()),
            'text': list(self.text_transformer.parameters()),
            'speech': list(self.speech_transformer.parameters()),
        }

    def forward(
            self,
            text,
            mel_input,
            mel_cond,
            return_loss=False
    ):
        device = text.device

        text_emb = self.text_emb(text)
        speech_emb = self.speech_emb(mel_input).permute(0,2,1)

        unused_params = []
        if random() < self.mask_conditioning_percentage:
            enc_cond = self.masked_conditioning_latent
            unused_params.extend(list(self.cond_emb.parameters()) + list(self.conditioning_transformer.parameters()))
        else:
            cond_emb = self.cond_emb(mel_cond).permute(0,2,1)
            enc_cond = self.conditioning_transformer(cond_emb)
            unused_params.append(self.masked_conditioning_latent)
        enc_text = self.text_transformer(text_emb, norm_scale_shift_inp=enc_cond)
        enc_speech = self.speech_transformer(speech_emb)

        text_latents = self.to_text_latent(enc_text)
        speech_latents = self.to_speech_latent(enc_speech)

        text_latents, speech_latents = map(lambda t: F.normalize(t, p=2, dim=-1), (text_latents, speech_latents))
        temp = self.temperature.exp()

        if self.distributed_collect:
            collective = [torch.zeros_like(text_latents) for _ in range(torch.distributed.get_world_size())]
            torch.all_gather(collective, text_latents)
            text_latents = torch.cat(collective, dim=0)
            collective = [torch.zeros_like(speech_latents) for _ in range(torch.distributed.get_world_size())]
            torch.all_gather(collective, speech_latents)
            speech_latents = torch.cat(collective, dim=0)

        if not return_loss:
            sim = einsum('n d, n d -> n', text_latents, speech_latents) * temp
            return sim

        sim = einsum('i d, j d -> i j', text_latents, speech_latents) * temp
        labels = torch.arange(text_latents.shape[0], device=device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        loss = loss + extraneous_addition * 0

        return loss


@register_model
def register_clvp(opt_net, opt):
    return CLVP(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    clvp = CLVP()
    clvp(torch.randint(0,256,(2,120)),
         torch.randn(2,80,100),
         torch.randn(2,80,95),
         return_loss=True)
    nonloss = clvp(torch.randint(0,256,(2,120)),
         torch.randn(2,80,100),
         torch.randn(2,80,95),
         return_loss=False)
    clvp = CLVP(mel_codes=8192)
    clvp(torch.randint(0,256,(2,120)),
         torch.randint(0,8192,(2,150)),
         torch.randn(2,80,95),
         return_loss=True)
    print(nonloss.shape)