import functools

import torch
from torch import nn
import torch.nn.functional as F

from models.arch_util import zero_module
from models.vqvae.vqvae import Quantize
from trainer.networks import register_model
from utils.util import checkpoint, ceil_multiple, print_network


class Downsample(nn.Module):
    def __init__(self, chan_in, chan_out):
        super().__init__()
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=.5, mode='linear')
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, chan_in, chan_out):
        super().__init__()
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='linear')
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, 3, padding = 1),
            nn.GroupNorm(8, chan),
            nn.SiLU(),
            nn.Conv1d(chan, chan, 3, padding = 1),
            nn.GroupNorm(8, chan),
            nn.SiLU(),
            zero_module(nn.Conv1d(chan, chan, 3, padding = 1)),
        )

    def forward(self, x):
        return checkpoint(self._forward, x) + x

    def _forward(self, x):
        return self.net(x)


class Wav2Vec2GumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See `[CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """

    def __init__(self, proj_dim=1024, codevector_dim=512, num_codevector_groups=2, num_codevectors_per_group=320):
        super().__init__()
        self.codevector_dim = codevector_dim
        self.num_groups = num_codevector_groups
        self.num_vars = num_codevectors_per_group
        self.num_codevectors = num_codevector_groups * num_codevectors_per_group

        if codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`codevector_dim {codevector_dim} must be divisible "
                f"by `num_codevector_groups` {num_codevector_groups} for concatenation"
            )

        # storage for codebook variables (codewords)
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, codevector_dim // self.num_groups)
        )
        self.weight_proj = nn.Linear(proj_dim, self.num_groups * self.num_vars)

        # can be decayed for training
        self.temperature = 2

        # Parameters init.
        self.weight_proj.weight.data.normal_(mean=0.0, std=1)
        self.weight_proj.bias.data.zero_()
        nn.init.uniform_(self.codevectors)

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)

        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity

    def get_codes(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)
        codevector_idx = hidden_states.argmax(dim=-1)
        idxs = codevector_idx.view(batch_size, sequence_length, self.num_groups)
        return idxs

    def forward(self, hidden_states, mask_time_indices=None, return_probs=False):
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # sample code vector probs via gumbel in differentiable way
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # compute perplexity
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # take argmax in non-differentiable way
            # compute hard codevector distribution (one hot)
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # use probs to retrieve codevectors
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = (
            codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
            .sum(-2)
            .view(batch_size, sequence_length, -1)
        )

        if return_probs:
            return codevectors, perplexity, codevector_probs.view(batch_size, sequence_length, self.num_groups, self.num_vars)
        return codevectors, perplexity


class MusicQuantizer2(nn.Module):
    def __init__(self, inp_channels=256, inner_dim=1024, codevector_dim=1024, down_steps=2,
                 max_gumbel_temperature=2.0, min_gumbel_temperature=.5, gumbel_temperature_decay=.999995,
                 codebook_size=16, codebook_groups=4):
        super().__init__()
        if not isinstance(inner_dim, list):
            inner_dim = [inner_dim // 2 ** x for x in range(down_steps+1)]
        self.max_gumbel_temperature = max_gumbel_temperature
        self.min_gumbel_temperature = min_gumbel_temperature
        self.gumbel_temperature_decay = gumbel_temperature_decay
        self.quantizer = Wav2Vec2GumbelVectorQuantizer(inner_dim[0], codevector_dim=codevector_dim,
                                                       num_codevector_groups=codebook_groups,
                                                       num_codevectors_per_group=codebook_size)
        self.codebook_size = codebook_size
        self.codebook_groups = codebook_groups
        self.num_losses_record = []

        if down_steps == 0:
            self.down = nn.Conv1d(inp_channels, inner_dim[0], kernel_size=3, padding=1)
            self.up = nn.Conv1d(inner_dim[0], inp_channels, kernel_size=3, padding=1)
        elif down_steps == 2:
            self.down = nn.Sequential(nn.Conv1d(inp_channels, inner_dim[-1], kernel_size=3, padding=1),
                                      *[Downsample(inner_dim[-i], inner_dim[-i-1]) for i in range(1,len(inner_dim))])
            self.up = nn.Sequential(*[Upsample(inner_dim[i], inner_dim[i+1]) for i in range(len(inner_dim)-1)] +
                                    [nn.Conv1d(inner_dim[-1], inp_channels, kernel_size=3, padding=1)])

        self.encoder = nn.Sequential(ResBlock(inner_dim[0]),
                                     ResBlock(inner_dim[0]),
                                     ResBlock(inner_dim[0]))
        self.enc_norm = nn.LayerNorm(inner_dim[0], eps=1e-5)
        self.decoder = nn.Sequential(nn.Conv1d(codevector_dim, inner_dim[0], kernel_size=3, padding=1),
                                     ResBlock(inner_dim[0]),
                                     ResBlock(inner_dim[0]),
                                     ResBlock(inner_dim[0]))

        self.codes = torch.zeros((3000000,), dtype=torch.long)
        self.internal_step = 0
        self.code_ind = 0
        self.total_codes = 0

    def get_codes(self, mel, project=False):
        proj = self.m2v.input_blocks(mel).permute(0,2,1)
        _, proj = self.m2v.projector(proj)
        if project:
            proj, _ = self.quantizer(proj)
            return proj
        else:
            return self.quantizer.get_codes(proj)

    def forward(self, mel, return_decoder_latent=False):
        orig_mel = mel
        cm = ceil_multiple(mel.shape[-1], 2 ** (len(self.down)-1))
        if cm != 0:
            mel = F.pad(mel, (0,cm-mel.shape[-1]))

        h = self.down(mel)
        h = self.encoder(h)
        h = self.enc_norm(h.permute(0,2,1))
        codevectors, perplexity, codes = self.quantizer(h, return_probs=True)
        diversity = (self.quantizer.num_codevectors - perplexity) / self.quantizer.num_codevectors
        self.log_codes(codes)
        h = self.decoder(codevectors.permute(0,2,1))
        if return_decoder_latent:
            return h, diversity

        reconstructed = self.up(h.float())
        reconstructed = reconstructed[:, :, :orig_mel.shape[-1]]

        mse = F.mse_loss(reconstructed, orig_mel)
        return mse, diversity

    def log_codes(self, codes):
        if self.internal_step % 5 == 0:
            codes = torch.argmax(codes, dim=-1)
            ccodes = codes[:,:,0]
            for j in range(1,codes.shape[-1]):
                ccodes += codes[:,:,j] * self.codebook_size ** j
            codes = ccodes
            codes = codes.flatten()
            l = codes.shape[0]
            i = self.code_ind if (self.codes.shape[0] - self.code_ind) > l else self.codes.shape[0] - l
            self.codes[i:i+l] = codes.cpu()
            self.code_ind = self.code_ind + l
            if self.code_ind >= self.codes.shape[0]:
                self.code_ind = 0
            self.total_codes += 1

    def get_debug_values(self, step, __):
        if self.total_codes > 0:
            return {'histogram_codes': self.codes[:self.total_codes]}
        else:
            return {}

    def update_for_step(self, step, *args):
        self.quantizer.temperature = max(
                    self.max_gumbel_temperature * self.gumbel_temperature_decay**step,
                    self.min_gumbel_temperature,
                )


@register_model
def register_music_quantizer2(opt_net, opt):
    return MusicQuantizer2(**opt_net['kwargs'])


if __name__ == '__main__':
    model = MusicQuantizer2(inner_dim=[1024], codevector_dim=1024, codebook_size=256, codebook_groups=2)
    print_network(model)
    mel = torch.randn((2,256,782))
    model(mel)