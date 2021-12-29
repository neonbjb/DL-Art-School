import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

from models.gpt_voice.mini_encoder import AudioMiniEncoder
from models.lucidrains.dalle.transformer import Transformer
from trainer.networks import register_model
from utils.util import opt_get


def exists(val):
    return val is not None


def masked_mean(t, mask, dim=1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]


class VoiceCLIP(nn.Module):
    """
    CLIP model modified to produce similarity scores from different views of the same audio clip.
    """

    def __init__(
            self,
            encoder_output=512,
            dim_latent=512,
            speech_max_seq_len=250,
            mel_compression_ratio=256,
            pretrained_encoder_dict_path=None
    ):
        super().__init__()
        self.encoder = AudioMiniEncoder(80, encoder_output)
        if pretrained_encoder_dict_path is not None:
            self.encoder.load_state_dict(torch.load(pretrained_encoder_dict_path))
        self.to_latent = nn.Linear(encoder_output, dim_latent, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.mel_compression_ratio = mel_compression_ratio

    def forward(
        self,
        speech_mels,
        speech_lengths,
        return_loss=True
    ):
        half_length = min(speech_mels.shape[-1], torch.min(speech_lengths).item() // self.mel_compression_ratio) // 2
        half_length = (half_length // 4) * 4  # Must be a multiple of 4.

        first_half = speech_mels[:, :, :half_length]
        second_half = speech_mels[:, :, half_length:half_length*2]

        first_emb = self.encoder(first_half)
        first_latents = self.to_latent(first_emb)
        second_emb = self.encoder(second_half)
        second_latents = self.to_latent(second_emb)

        first_latents, second_latents = map(lambda t: F.normalize(t, p=2, dim=-1), (first_latents, second_latents))

        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum('n d, n d -> n', first_latents, second_latents) * temp
            return sim

        sim = einsum('i d, j d -> i j', first_latents, second_latents) * temp
        labels = torch.arange(first_latents.shape[0], device=first_latents.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss

    def inference(self, speech_mels):
        emb = self.encoder(speech_mels)
        latent = self.to_latent(emb)
        latent = F.normalize(latent, p=2, dim=-1)
        temp = self.temperature.exp()
        sim = einsum('i d, j d -> i j', latent, latent) * temp
        return sim


@register_model
def register_voice_to_voice_clip(opt_net, opt):
    return VoiceCLIP(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    clip = VoiceCLIP()
    clip(torch.randn((2,80,200)),
         torch.randint(0,200*1024,(2,)),
         return_loss=True)