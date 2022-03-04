import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

from models.lucidrains.dalle.transformer import Transformer
from trainer.networks import register_model
from utils.util import opt_get


def exists(val):
    return val is not None


def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]


class MelTextCLIP(nn.Module):
    """
    CLIP model retrofitted for performing contrastive evaluation between MEL data and the corresponding
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
            mel_compression=256,
    ):
        super().__init__()
        self.text_emb = nn.Embedding(num_text_tokens, dim_text)
        self.text_pos_emb = nn.Embedding(text_seq_len, dim_text)
        self.text_transformer = Transformer(causal=False, seq_len=text_seq_len, dim=dim_text, depth=text_enc_depth,
                                            heads=text_heads, rotary_emb=False)
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias=False)

        self.speech_enc = nn.Conv1d(80, dim_speech, kernel_size=3, padding=1)
        self.speech_pos_emb = nn.Embedding(num_speech_tokens, dim_speech)
        self.speech_transformer = Transformer(causal=False, seq_len=speech_seq_len, dim=dim_speech,
                                              depth=speech_enc_depth, heads=speech_heads, rotary_emb=False)
        self.to_speech_latent = nn.Linear(dim_speech, dim_latent, bias=False)

        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_mask_percentage = text_mask_percentage
        self.voice_mask_percentage = voice_mask_percentage
        self.mel_compression = mel_compression

    def get_text_projections(self, text, text_mask=None):
        if text_mask is None:
            text_mask = torch.ones_like(text.float()).bool()
        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device=text.device))
        with torch.autocast(text.device.type):
            enc_text = self.text_transformer(text_emb, mask=text_mask)
            text_latents = masked_mean(enc_text, text_mask, dim=1)
        return self.to_text_latent(text_latents).float()

    def get_speech_projection(self, mel, voice_mask=None):
        if voice_mask is None:
            voice_mask = torch.ones_like(mel[:,0,:].float()).bool()
        speech_emb = self.speech_enc(mel).permute(0,2,1)
        speech_emb += self.speech_pos_emb(torch.arange(speech_emb.shape[1], device=mel.device))
        with torch.autocast(speech_emb.device.type):
            enc_speech = self.speech_transformer(speech_emb, mask=voice_mask)
            speech_latents = masked_mean(enc_speech, voice_mask, dim=1)
        return self.to_speech_latent(speech_latents).float()

    def forward(
            self,
            text,
            text_lengths,
            mel,
            wav_lengths,
            return_loss=False
    ):
        # This model will receive micro-batches with a ton of padding for both the text and MELs. Ameliorate this by
        # chopping the inputs by the maximum actual length.
        max_text_len = text_lengths.max()
        text = text[:, :max_text_len]
        max_mel_len = wav_lengths.max() // self.mel_compression
        mel = mel[:, :, :max_mel_len]

        b, device = text.shape[0], text.device
        if self.training:
            text_mask = torch.rand_like(text.float()) > self.text_mask_percentage
            voice_mask = torch.rand_like(mel[:,0,:].float()) > self.voice_mask_percentage
        else:
            text_mask = torch.ones_like(text.float()).bool()
            voice_mask = torch.ones_like(mel[:,0,:].float()).bool()

        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device=device))

        speech_emb = self.speech_enc(mel).permute(0,2,1)
        speech_emb += self.speech_pos_emb(torch.arange(speech_emb.shape[1], device=device))

        # Only autocast the transformer part. The MEL encoder loses accuracy if you autcast it.
        with torch.autocast(speech_emb.device.type):
            enc_text = self.text_transformer(text_emb, mask=text_mask)
            enc_speech = self.speech_transformer(speech_emb, mask=voice_mask)

            text_latents = masked_mean(enc_text, text_mask, dim=1)
            speech_latents = masked_mean(enc_speech, voice_mask, dim=1)

        text_latents = self.to_text_latent(text_latents).float()
        speech_latents = self.to_speech_latent(speech_latents).float()

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
def register_mel_text_clip(opt_net, opt):
    return MelTextCLIP(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    clip = MelTextCLIP(text_mask_percentage=.2, voice_mask_percentage=.2)
    clip(torch.randint(0,256,(2,120)),
         torch.tensor([50,100]),
         torch.randn(2,80,400),
         torch.tensor([10100,10200]),
         return_loss=True)