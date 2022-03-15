import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from models.audio.tts.mini_encoder import AudioMiniEncoder
from trainer.injectors.spec_augment import spec_augment
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

        # Extract two speech MELs from the same clip, apply some random noise to them and also apply specaugment to them.
        first_half = speech_mels[:, :, :half_length]
        first_half = first_half + torch.rand_like(first_half) * .00001
        first_half = spec_augment(first_half)
        second_half = speech_mels[:, :, half_length:half_length*2]
        second_half = second_half + torch.rand_like(second_half) * .00001
        second_half = spec_augment(second_half)

        # Introduce a random gap between the two clips.
        potential_gap = half_length // 4
        gap = random.randint(0, potential_gap)
        if gap > 0:
            first_half = first_half[:, :, :-gap]
            second_half = second_half[:, :, gap:]

        # The clips must be multiples of 4.
        if first_half.shape[-1] % 4 != 0:
            first_half = first_half[:, :, :first_half.shape[-1] // 4 * 4]
        if second_half.shape[-1] % 4 != 0:
            second_half = second_half[:, :, :second_half.shape[-1] // 4 * 4]

        # Flip the clips randomly
        if random.random() < .5:
            t = first_half
            first_half = second_half
            second_half = t

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
    for k in range(1000):
        clip(torch.randn((2,80,156)),
             torch.randint(130*1024,156*1024,(2,)),
             return_loss=True)