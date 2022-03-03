import functools
import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

from models.vqvae.vector_quantizer import VectorQuantize
from models.vqvae.vqvae import Quantize
from trainer.networks import register_model
from utils.util import opt_get


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class ResBlock(nn.Module):
    def __init__(self, chan, conv, activation):
        super().__init__()
        self.net = nn.Sequential(
            conv(chan, chan, 3, padding = 1),
            activation(),
            conv(chan, chan, 3, padding = 1),
            activation(),
            conv(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class UpsampledConv(nn.Module):
    def __init__(self, conv, *args, **kwargs):
        super().__init__()
        assert 'stride' in kwargs.keys()
        self.stride = kwargs['stride']
        del kwargs['stride']
        self.conv = conv(*args, **kwargs)

    def forward(self, x):
        up = nn.functional.interpolate(x, scale_factor=self.stride, mode='nearest')
        return self.conv(up)


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
        stride = 2,
        kernel_size = 3,
        activation = 'relu',
        straight_through = False,
        record_codes = False,
        discretization_loss_averaging_steps = 100,
        quantizer_use_cosine_sim=True,
        quantizer_codebook_misses_to_expiration=40,
        quantizer_codebook_embedding_compression=None,
    ):
        super().__init__()
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.straight_through = straight_through
        self.positional_dims = positional_dims

        assert positional_dims > 0 and positional_dims < 3  # This VAE only supports 1d and 2d inputs for now.
        if positional_dims == 2:
            conv = nn.Conv2d
            conv_transpose = functools.partial(UpsampledConv, conv)
        else:
            conv = nn.Conv1d
            conv_transpose = functools.partial(UpsampledConv, conv)

        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'silu':
            act = nn.SiLU
        else:
            assert NotImplementedError()


        enc_chans = [hidden_dim * 2 ** i for i in range(num_layers)]
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        pad = (kernel_size - 1) // 2
        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(conv(enc_in, enc_out, kernel_size, stride = stride, padding = pad), act()))
            dec_layers.append(nn.Sequential(conv_transpose(dec_in, dec_out, kernel_size, stride = stride, padding = pad), act()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1], conv, act))
            enc_layers.append(ResBlock(enc_chans[-1], conv, act))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, conv(codebook_dim, dec_chans[1], 1))

        enc_layers.append(conv(enc_chans[-1], codebook_dim, 1))
        dec_layers.append(conv(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.quantizer = VectorQuantize(codebook_dim, num_tokens, codebook_dim=quantizer_codebook_embedding_compression,
                                        use_cosine_sim=quantizer_use_cosine_sim,
                                        max_codebook_misses_before_expiry=quantizer_codebook_misses_to_expiration)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.mse_loss

        self.record_codes = record_codes
        if record_codes:
            self.codes = torch.zeros((1228800,), dtype=torch.long)
            self.code_ind = 0
        self.internal_step = 0

    def get_debug_values(self, step, __):
        if self.record_codes:
            # Report annealing schedule
            return {'histogram_codes': self.codes}
        else:
            return {}

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self.encoder(images).permute((0,2,3,1) if len(images.shape) == 4 else (0,2,1))
        sampled, codes, commitment_loss = self.quantizer(logits)
        return codes

    def decode(
        self,
        img_seq
    ):
        self.log_codes(img_seq)
        image_embeds = self.quantizer.decode(img_seq)
        b, n, d = image_embeds.shape

        kwargs = {}
        if self.positional_dims == 1:
            arrange = 'b n d -> b d n'
        else:
            h = w = int(sqrt(n))
            arrange = 'b (h w) d -> b d h w'
            kwargs = {'h': h, 'w': w}
        image_embeds = rearrange(image_embeds, arrange, **kwargs)
        images = [image_embeds]
        for layer in self.decoder:
            images.append(layer(images[-1]))
        return images[-1], images[-2]

    def infer(self, img):
        logits = self.encoder(img).permute((0,2,3,1) if len(img.shape) == 4 else (0,2,1))
        sampled, codes, commitment_loss = self.quantizer(logits)
        return self.decode(codes)

    # Note: This module is not meant to be run in forward() except while training. It has special logic which performs
    # evaluation using quantized values when it detects that it is being run in eval() mode, which will be substantially
    # more lossy (but useful for determining network performance).
    def forward(
        self,
        img
    ):
        logits = self.encoder(img).permute((0,2,3,1) if len(img.shape) == 4 else (0,2,1))
        sampled, codes, commitment_loss = self.quantizer(logits)
        sampled = sampled.permute((0,3,1,2) if len(img.shape) == 4 else (0,2,1))

        if self.training:
            out = sampled
            for d in self.decoder:
                out = d(out)
        else:
            # This is non-differentiable, but gives a better idea of how the network is actually performing.
            out, _ = self.decode(codes)

        # reconstruction loss
        recon_loss = self.loss_fn(img, out, reduction='none')

        # This is so we can debug the distribution of codes being learned.
        self.log_codes(codes)

        return recon_loss, commitment_loss, out

    def log_codes(self, codes):
        # This is so we can debug the distribution of codes being learned.
        if self.record_codes and self.internal_step % 50 == 0:
            codes = codes.flatten()
            l = codes.shape[0]
            i = self.code_ind if (self.codes.shape[0] - self.code_ind) > l else self.codes.shape[0] - l
            self.codes[i:i+l] = codes.cpu()
            self.code_ind = self.code_ind + l
            if self.code_ind >= self.codes.shape[0]:
                self.code_ind = 0
        self.internal_step += 1


@register_model
def register_dvae(opt_net, opt):
    return DiscreteVAE(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    #v = DiscreteVAE()
    #o=v(torch.randn(1,3,256,256))
    #print(o.shape)
    v = DiscreteVAE(channels=80, positional_dims=1, num_tokens=4096, codebook_dim=1024,
                    hidden_dim=512, stride=2, num_resnet_blocks=2, kernel_size=3, num_layers=2,
                    quantizer_codebook_embedding_compression=64)
    #v.eval()
    loss, commitment, out = v(torch.randn(1,80,256))
    print(out.shape)
    codes = v.get_codebook_indices(torch.randn(1,80,256))
    back, back_emb = v.decode(codes)
    print(back.shape)
