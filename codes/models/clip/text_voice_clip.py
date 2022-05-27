from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from x_transformers import Encoder

from models.audio.tts.unet_diffusion_tts7 import CheckpointedXTransformerEncoder
from models.lucidrains.dalle.transformer import Transformer
from trainer.networks import register_model
from utils.util import opt_get


def exists(val):
    return val is not None


def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]


class VoiceCLIP(nn.Module):
    """
    CLIP model retrofitted for performing contrastive evaluation between tokenized audio data and the corresponding
    transcribed text.

    Originally from https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
    """

    def __init__(
            self,
            *,
            dim_text=512,
            dim_speech=512,
            dim_latent=512,
            num_text_tokens=256,
            text_enc_depth=6,
            text_seq_len=120,
            text_heads=8,
            num_speech_tokens=8192,
            speech_enc_depth=6,
            speech_heads=8,
            speech_seq_len=250,
            text_mask_percentage=0,
            voice_mask_percentage=0,
            wav_token_compression=1024,
            use_xformers=False,
            clip_mels=False,
            min_mel_size=10,  # Default is approximately .5sec with default mel specs.
            distributed_collect=False,
    ):
        super().__init__()
        self.text_emb = nn.Embedding(num_text_tokens, dim_text)
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias=False)

        self.speech_emb = nn.Embedding(num_speech_tokens, dim_speech)
        self.to_speech_latent = nn.Linear(dim_speech, dim_latent, bias=False)

        if use_xformers:
            self.text_transformer = CheckpointedXTransformerEncoder(
                needs_permute=False,
                exit_permute=False,
                max_seq_len=-1,
                attn_layers=Encoder(
                    dim=dim_text,
                    depth=text_enc_depth,
                    heads=text_heads,
                    ff_dropout=.1,
                    ff_mult=2,
                    attn_dropout=.1,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                ))
            self.speech_transformer = CheckpointedXTransformerEncoder(
                needs_permute=False,
                exit_permute=False,
                max_seq_len=-1,
                attn_layers=Encoder(
                    dim=dim_speech,
                    depth=speech_enc_depth,
                    heads=speech_heads,
                    ff_dropout=.1,
                    ff_mult=2,
                    attn_dropout=.1,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                ))
        else:
            self.text_transformer = Transformer(causal=False, seq_len=text_seq_len, dim=dim_text, depth=text_enc_depth,
                                                heads=text_heads)
            self.speech_transformer = Transformer(causal=False, seq_len=speech_seq_len, dim=dim_speech,
                                                  depth=speech_enc_depth, heads=speech_heads)

        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_mask_percentage = text_mask_percentage
        self.voice_mask_percentage = voice_mask_percentage
        self.wav_token_compression = wav_token_compression
        self.xformers = use_xformers
        self.clip_mels = clip_mels
        self.min_mel_size = min_mel_size
        self.distributed_collect = distributed_collect
        if not use_xformers:
            self.text_pos_emb = nn.Embedding(text_seq_len, dim_text)
            self.speech_pos_emb = nn.Embedding(num_speech_tokens, dim_speech)

    def embed_text(self, text):
        text_mask = torch.ones_like(text.float()).bool()
        text_emb = self.text_emb(text)
        enc_text = self.text_transformer(text_emb, mask=text_mask)
        text_latents = masked_mean(enc_text, text_mask, dim=1)
        text_latents = self.to_text_latent(text_latents)
        return text_latents

    def forward(
            self,
            text,
            speech_tokens,
            return_loss=False
    ):
        b, device = text.shape[0], text.device
        if self.training:
            if self.clip_mels:
                margin = speech_tokens.shape[-1] - self.min_mel_size
                speech_tokens = speech_tokens[:, :self.min_mel_size+randint(0,margin)]
                voice_mask = torch.ones_like(speech_tokens.float()).bool()  # Disable voice masking in this case.
            else:
                voice_mask = torch.rand_like(speech_tokens.float()) > self.voice_mask_percentage
            text_mask = torch.rand_like(text.float()) > self.text_mask_percentage
        else:
            text_mask = torch.ones_like(text.float()).bool()
            voice_mask = torch.ones_like(speech_tokens.float()).bool()

        text_emb = self.text_emb(text)
        speech_emb = self.speech_emb(speech_tokens)

        if not self.xformers:
            text_emb += self.text_pos_emb(torch.arange(text.shape[1], device=device))
            speech_emb += self.speech_pos_emb(torch.arange(speech_emb.shape[1], device=device))

        enc_text = self.text_transformer(text_emb, mask=text_mask)
        enc_speech = self.speech_transformer(speech_emb, mask=voice_mask)

        text_latents = masked_mean(enc_text, text_mask, dim=1)
        speech_latents = masked_mean(enc_speech, voice_mask, dim=1)

        text_latents = self.to_text_latent(text_latents)
        speech_latents = self.to_speech_latent(speech_latents)

        if self.distributed_collect:
            collective = [torch.zeros_like(text_latents) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(collective, text_latents)
            collective[torch.distributed.get_rank()] = text_latents  # For gradient propagation.
            text_latents = torch.cat(collective, dim=0)
            collective = [torch.zeros_like(speech_latents) for _ in range(torch.distributed.get_world_size())]
            collective[torch.distributed.get_rank()] = speech_latents  # For gradient propagation.            
            torch.distributed.all_gather(collective, speech_latents)
            speech_latents = torch.cat(collective, dim=0)
            b = text_latents.shape[0]

        text_latents, speech_latents = map(lambda t: F.normalize(t, p=2, dim=-1), (text_latents, speech_latents))

        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum('n d, n d -> n', text_latents, speech_latents) * temp
            return sim

        sim = einsum('i d, j d -> i j', text_latents, speech_latents) * temp
        labels = torch.arange(b, device=device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss


@register_model
def register_voice_clip(opt_net, opt):
    return VoiceCLIP(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    clip = VoiceCLIP(text_mask_percentage=.2, voice_mask_percentage=.2, use_xformers=True)
    clip(torch.randint(0,256,(2,120)),
         torch.randint(0,8192,(2,250)),
         return_loss=True)
    nonloss = clip(torch.randint(0,256,(2,120)),
         torch.randint(0,8192,(2,250)),
         return_loss=False)
    print(nonloss.shape)
