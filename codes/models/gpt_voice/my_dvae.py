import functools
import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

from models.diffusion.nn import conv_nd, normalization, zero_module
from models.diffusion.unet_diffusion import Upsample, Downsample, AttentionBlock
from models.vqvae.vqvae import Quantize
from trainer.networks import register_model
from utils.util import opt_get, checkpoint


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
    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        kernel_size=3,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = 1 if kernel_size == 3 else 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        return checkpoint(
            self._forward, x
        )

    def _forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class DisjointUnet(nn.Module):
    def __init__(
            self,
            attention_resolutions,
            channel_mult_down,
            channel_mult_up,
            in_channels = 3,
            model_channels = 64,
            out_channels = 3,
            dims=2,
            num_res_blocks = 2,
            stride = 2,
            dropout=0,
            num_heads=4,
    ):
        super().__init__()

        self.enc_input_blocks = nn.ModuleList(
            [
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult_down):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=-1,
                        )
                    )
                self.enc_input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult_down) - 1:
                out_ch = ch
                self.enc_input_blocks.append(
                    Downsample(
                        ch, True, dims=dims, out_channels=out_ch, factor=stride
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        self.enc_middle_block = nn.Sequential(
            ResBlock(
                ch,
                dropout,
                dims=dims,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=-1,
            ),
            ResBlock(
                ch,
                dropout,
                dims=dims,
            ),
        )

        self.enc_output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult_up)):
            for i in range(num_res_blocks + 1):
                if len(input_block_chans) > 0:
                    ich = input_block_chans.pop()
                else:
                    ich = 0
                layers = [
                    ResBlock(
                        ch + ich,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=-1,
                        )
                    )
                if level != len(channel_mult_up)-1 and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        Upsample(ch, True, dims=dims, out_channels=out_ch, factor=stride)
                    )
                    ds //= 2
                self.enc_output_blocks.append(nn.Sequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, ch, out_channels, 3, padding=1),
        )

    def forward(self, x):
        hs = []
        h = x
        for module in self.enc_input_blocks:
            h = module(h)
            hs.append(h)
        h = self.enc_middle_block(h)
        for module in self.enc_output_blocks:
            if len(hs) > 0:
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h)
        h = h.type(x.dtype)
        return self.out(h)


class DiscreteVAE(nn.Module):
    def __init__(
        self,
        attention_resolutions,
        in_channels = 3,
        model_channels = 64,
        out_channels = 3,
        channel_mult=(1, 2, 4, 8),
        dims=2,
        num_tokens = 512,
        codebook_dim = 512,
        convergence_layer=2,
        num_res_blocks = 0,
        stride = 2,
        straight_through = False,
        dropout=0,
        num_heads=4,
        record_codes=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_tokens = num_tokens
        self.num_layers = len(channel_mult)
        self.straight_through = straight_through
        self.codebook = Quantize(codebook_dim, num_tokens)
        self.positional_dims = dims
        self.dropout = dropout
        self.num_heads = num_heads
        self.record_codes = record_codes
        if record_codes:
            self.codes = torch.zeros((32768,), dtype=torch.long)
            self.code_ind = 0
        self.internal_step = 0

        enc_down = channel_mult
        enc_up = list(reversed(channel_mult[convergence_layer:]))
        self.encoder = DisjointUnet(attention_resolutions, enc_down, enc_up, in_channels=in_channels, model_channels=model_channels,
                                    out_channels=codebook_dim, dims=dims, num_res_blocks=num_res_blocks, num_heads=num_heads, dropout=dropout,
                                    stride=stride)
        dec_down = list(reversed(enc_up))
        dec_up = list(reversed(enc_down))
        self.decoder = DisjointUnet(attention_resolutions, dec_down, dec_up, in_channels=codebook_dim, model_channels=model_channels,
                                    out_channels=out_channels, dims=dims, num_res_blocks=num_res_blocks, num_heads=num_heads, dropout=dropout,
                                    stride=stride)

    def get_debug_values(self, step, __):
        if self.record_codes:
            # Report annealing schedule
            return {'histogram_codes': self.codes}
        else:
            return {}

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        img = images
        logits = self.encoder(img).permute((0,2,3,1) if len(img.shape) == 4 else (0,2,1))
        sampled, commitment_loss, codes = self.codebook(logits)
        return codes

    def decode(
        self,
        img_seq
    ):
        image_embeds = self.codebook.embed_code(img_seq)
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

    def infer(self, img):
        logits = self.encoder(img).permute((0,2,3,1) if len(img.shape) == 4 else (0,2,1))
        sampled, commitment_loss, codes = self.codebook(logits)
        return self.decode(codes)

    # Note: This module is not meant to be run in forward() except while training. It has special logic which performs
    # evaluation using quantized values when it detects that it is being run in eval() mode, which will be substantially
    # more lossy (but useful for determining network performance).
    def forward(
        self,
        img
    ):
        logits = self.encoder(img).permute((0,2,3,1) if len(img.shape) == 4 else (0,2,1))
        sampled, commitment_loss, codes = self.codebook(logits)
        sampled = sampled.permute((0,3,1,2) if len(img.shape) == 4 else (0,2,1))

        if self.training:
            out = sampled
            out = self.decoder(out)
        else:
            # This is non-differentiable, but gives a better idea of how the network is actually performing.
            out = self.decode(codes)

        # reconstruction loss
        recon_loss = F.mse_loss(img, out, reduction='none')

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

        return recon_loss, commitment_loss, out


@register_model
def register_my_dvae(opt_net, opt):
    return DiscreteVAE(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    net = DiscreteVAE((8, 16), channel_mult=(1,2,4,8,8), in_channels=80, model_channels=128, out_channels=80, dims=1, num_res_blocks=2)
    inp = torch.randn((2,80,512))
    print([j.shape for j in net(inp)])
