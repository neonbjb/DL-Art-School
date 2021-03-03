import torch
from kornia import filter2D
from torch import nn
from torch.nn import functional as F

import torch.distributed as distributed

from models.vqvae.vqvae import ResBlock, Quantize
from trainer.networks import register_model
from utils.util import checkpoint, opt_get


# Upsamples and blurs (similar to StyleGAN). Replaces ConvTranspose2D from the original paper.
class UpsampleConv(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size, padding=padding)

    def forward(self, x):
        up = torch.nn.functional.interpolate(x, scale_factor=2)
        return self.conv(up)


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 5, stride=2, padding=2),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 5, stride=2, padding=2),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 5, stride=2, padding=2),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.LeakyReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.LeakyReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    UpsampleConv(channel, channel // 2, 5, padding=2),
                    nn.LeakyReLU(inplace=True),
                    UpsampleConv(
                        channel // 2, out_channel, 5, padding=2
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                UpsampleConv(channel, out_channel, 5, padding=2)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE3(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        codebook_dim=64,
        codebook_size=512,
        decay=0.99,
    ):
        super().__init__()

        self.initial_conv = nn.Sequential(*[nn.Conv2d(in_channel, 32, 3, padding=1),
                                           nn.LeakyReLU(inplace=True)])
        self.enc_b = Encoder(32, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, codebook_dim, 1)
        self.quantize_t = Quantize(codebook_dim, codebook_size)
        self.dec_t = Decoder(
            codebook_dim, codebook_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(codebook_dim + channel, codebook_dim, 1)
        self.quantize_b = Quantize(codebook_dim, codebook_size)
        self.upsample_t = UpsampleConv(
            codebook_dim, codebook_dim, 5, padding=2
        )
        self.dec = Decoder(
            codebook_dim + codebook_dim,
            32,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.final_conv = nn.Conv2d(32, in_channel, 3, padding=1)

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        fea = self.initial_conv(input)
        enc_b = checkpoint(self.enc_b, fea)
        enc_t = checkpoint(self.enc_t, enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = checkpoint(self.dec_t, quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = checkpoint(self.quantize_conv_b, enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = checkpoint(self.dec, quant)
        dec = checkpoint(self.final_conv, dec)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


@register_model
def register_vqvae3(opt_net, opt):
    kw = opt_get(opt_net, ['kwargs'], {})
    return VQVAE3(**kw)


if __name__ == '__main__':
    v = VQVAE3()
    print(v(torch.randn(1,3,128,128))[0].shape)
