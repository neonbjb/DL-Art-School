import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

from trainer.networks import register_model
from utils.util import opt_get


def default(val, d):
    return val if val is not None else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class ResBlock(nn.Module):
    def __init__(self, chan, conv):
        super().__init__()
        self.net = nn.Sequential(
            conv(chan, chan, 3, padding = 1),
            nn.ReLU(),
            conv(chan, chan, 3, padding = 1),
            nn.ReLU(),
            conv(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class DiscreteVAE(nn.Module):
    def __init__(
        self,
        positional_dims=2,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        num_resnet_blocks = 0,
        hidden_dim = 64,
        channels = 3,
        smooth_l1_loss = False,
        starting_temperature = 0.5,
        temperature_annealing_rate = 0,
        min_temperature = .5,
        straight_through = False,
        normalization = None, # ((0.5,) * 3, (0.5,) * 3),
        record_codes = False,
    ):
        super().__init__()
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.starting_temperature = starting_temperature
        self.current_temperature = starting_temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)
        self.positional_dims = positional_dims
        self.temperature_annealing_rate = temperature_annealing_rate
        self.min_temperature = min_temperature

        assert positional_dims > 0 and positional_dims < 3  # This VAE only supports 1d and 2d inputs for now.
        if positional_dims == 2:
            conv = nn.Conv2d
            conv_transpose = nn.ConvTranspose2d
        else:
            conv = nn.Conv1d
            conv_transpose = nn.ConvTranspose1d

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(conv(enc_in, enc_out, 4, stride = 2, padding = 1), nn.ReLU()))
            dec_layers.append(nn.Sequential(conv_transpose(dec_in, dec_out, 4, stride = 2, padding = 1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1], conv))
            enc_layers.append(ResBlock(enc_chans[-1], conv))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, conv(codebook_dim, dec_chans[1], 1))

        enc_layers.append(conv(enc_chans[-1], num_tokens, 1))
        dec_layers.append(conv(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

        # take care of normalization within class
        self.normalization = normalization
        self.record_codes = record_codes
        if record_codes:
            self.codes = torch.zeros((32768,), dtype=torch.long)
            self.code_ind = 0

    def norm(self, images):
        if not self.normalization is not None:
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        arrange = 'c -> () c () ()' if self.positional_dims == 2 else 'c -> () c ()'
        means, stds = map(lambda t: rearrange(t, arrange), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    def update_for_step(self, step, __):
        # Run the annealing schedule
        if self.temperature_annealing_rate != 0:
            self.current_temperature = max(self.starting_temperature * math.exp(-self.temperature_annealing_rate * step), self.min_temperature)

    def get_debug_values(self, step, __):
        # Report annealing schedule
        return {'current_annealing_temperature': self.current_temperature, 'histogram_codes': self.codes}

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits = True)
        codebook_indices = logits.argmax(dim = 1).flatten(1)
        return codebook_indices

    def decode(
        self,
        img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape

        kwargs = {}
        if self.positional_dims == 1:
            arrange = 'b n d -> b d n'
        else:
            h = w = int(sqrt(n))
            arrange = 'b (h w) d -> b d h w'
            kwargs = {'h': h, 'w': w}
        image_embeds = rearrange(image_embeds, arrange, **kwargs)
        images = self.decoder(image_embeds)
        return images

    def forward(
        self,
        img
    ):
        device, num_tokens = img.device, self.num_tokens
        img = self.norm(img)
        logits = self.encoder(img)
        soft_one_hot = F.gumbel_softmax(logits, tau = self.current_temperature, dim = 1, hard = self.straight_through)

        if self.positional_dims == 1:
            arrange = 'b n s, n d -> b d s'
        else:
            arrange = 'b n h w, n d -> b d h w'
        sampled = einsum(arrange, soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        # reconstruction loss
        recon_loss = self.loss_fn(img, out)

        # kl divergence
        arrange = 'b n h w -> b (h w) n' if self.positional_dims == 2 else 'b n s -> b s n'
        logits = rearrange(logits, arrange)
        log_qy = F.log_softmax(logits, dim = -1)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device = device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)

        # This is so we can debug the distribution of codes being learned.
        if self.record_codes:
            codes = logits.argmax(dim = 2).flatten()
            l = codes.shape[0]
            i = self.code_ind if (self.codes.shape[0] - self.code_ind) > l else self.codes.shape[0] - l
            self.codes[i:i+l] = codes.cpu()
            self.code_ind = self.code_ind + l
            if self.code_ind >= self.codes.shape[0]:
                self.code_ind = 0

        return recon_loss, kl_div, out


@register_model
def register_lucidrains_dvae(opt_net, opt):
    return DiscreteVAE(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    #v = DiscreteVAE()
    #o=v(torch.randn(1,3,256,256))
    #print(o.shape)
    v = DiscreteVAE(channels=1, normalization=None, positional_dims=1)
    o=v(torch.randn(1,1,256))
    print(o.shape)
