import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from models.audio.tts.unified_voice2 import ConditioningEncoder
from models.lucidrains.dalle.transformer import Transformer
from trainer.networks import register_model
from utils.util import opt_get


def exists(val):
    return val is not None


def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]


class VoiceCondCLIP(nn.Module):
    """
    CLIP model retrofitted for performing contrastive evaluation between tokenized audio data and an encoded conditioning
    clip.

    Originally from https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
    """

    def __init__(
            self,
            *,
            dim_speech=512,
            dim_latent=512,
            num_speech_tokens=8192,
            speech_enc_depth=6,
            speech_heads=8,
            speech_seq_len=250,
            voice_mask_percentage=0,
            wav_token_compression=1024,
    ):
        super().__init__()
        self.cond_encoder = ConditioningEncoder(80, dim_latent, do_checkpointing=True)

        self.speech_emb = nn.Embedding(num_speech_tokens, dim_speech)
        self.speech_pos_emb = nn.Embedding(num_speech_tokens, dim_speech)
        self.speech_transformer = Transformer(causal=False, seq_len=speech_seq_len, dim=dim_speech,
                                              depth=speech_enc_depth, heads=speech_heads, rotary_emb=False)
        self.to_speech_latent = nn.Linear(dim_speech, dim_latent, bias=False)

        self.temperature = nn.Parameter(torch.tensor(1.))
        self.voice_mask_percentage = voice_mask_percentage
        self.wav_token_compression = wav_token_compression

    def forward(
            self,
            cond_mel,
            speech_tokens,
            wav_lengths,
            return_loss=False
    ):
        # This model will receive micro-batches with a ton of padding for the speech tokens. Ameliorate this by
        # chopping the inputs by the maximum actual length.
        max_mel_len = wav_lengths.max() // self.wav_token_compression
        speech_tokens = speech_tokens[:, :max_mel_len]

        b, device = speech_tokens.shape[0], speech_tokens.device
        if self.training:
            voice_mask = torch.rand_like(speech_tokens.float()) > self.voice_mask_percentage
        else:
            voice_mask = torch.ones_like(speech_tokens.float()).bool()

        speech_emb = self.speech_emb(speech_tokens)
        speech_emb += self.speech_pos_emb(torch.arange(speech_emb.shape[1], device=device))

        cond_latents = self.cond_encoder(cond_mel)

        enc_speech = self.speech_transformer(speech_emb, mask=voice_mask)
        speech_latents = masked_mean(enc_speech, voice_mask, dim=1)
        speech_latents = self.to_speech_latent(speech_latents)

        cond_latents, speech_latents = map(lambda t: F.normalize(t, p=2, dim=-1), (cond_latents, speech_latents))

        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum('n d, n d -> n', cond_latents, speech_latents) * temp
            return sim

        sim = einsum('i d, j d -> i j', cond_latents, speech_latents) * temp
        labels = torch.arange(b, device=device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss


@register_model
def register_voice_cond_clip(opt_net, opt):
    return VoiceCondCLIP(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    clip = VoiceCondCLIP(voice_mask_percentage=.2)
    clip(torch.randn(2,80,400),
         torch.randint(0,8192,(2,250)),
         torch.tensor([101,102]),
         return_loss=True)
    nonloss = clip(
         torch.randn(2, 80, 400),
         torch.randint(0,8192,(2,250)),
         torch.tensor([101,102]),
         return_loss=False)
    print(nonloss.shape)