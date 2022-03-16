import math

import torch.nn as nn
import torch

from models.image_generation.RRDBNet_arch import RRDB
from models.arch_util import ConvGnLelu


# Produces a convolutional feature (`f`) and a reduced feature map with double the filters.
from models.image_generation.glean.stylegan2_latent_bank import Stylegan2LatentBank
from models.image_generation.stylegan.stylegan2_rosinality import EqualLinear
from trainer.networks import register_model
from utils.util import checkpoint, sequential_checkpoint


class GleanEncoderBlock(nn.Module):
    def __init__(self, nf, max_nf):
        super().__init__()
        self.structural_latent_conv = ConvGnLelu(nf, nf, kernel_size=1, activation=False, norm=False, bias=True)
        top_nf = min(nf*2, max_nf)
        self.process = nn.Sequential(
            ConvGnLelu(nf, top_nf, kernel_size=3, stride=2, activation=True, norm=False, bias=False),
            ConvGnLelu(top_nf, top_nf, kernel_size=3, activation=True, norm=False, bias=False)
        )

    def forward(self, x):
        structural_latent = self.structural_latent_conv(x)
        fea = self.process(x)
        return fea, structural_latent


# Produces RRDB features, a list of convolutional features (`f` shape=[l][b,c,h,w] l=levels aka f_sub)
# and latent vectors (`C` shape=[b,l,f] l=levels aka C_sub) for use with the latent bank.
# Note that latent levels and convolutional feature levels do not necessarily match, per the paper.
class GleanEncoder(nn.Module):
    def __init__(self, nf, nb, max_nf=512, reductions=4, latent_bank_blocks=7, latent_bank_latent_dim=512, input_dim=32, initial_stride=1):
        super().__init__()
        self.initial_conv = ConvGnLelu(3, nf, kernel_size=7, activation=False, norm=False, bias=True, stride=initial_stride)
        self.rrdb_blocks = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        self.reducers = nn.ModuleList([GleanEncoderBlock(min(nf * 2 ** i, max_nf), max_nf) for i in range(reductions)])

        reducer_output_dim = (input_dim // (2 ** (reductions + 1))) ** 2
        reducer_output_nf = min(nf * 2 ** reductions, max_nf)
        self.latent_conv = ConvGnLelu(reducer_output_nf, reducer_output_nf, stride=2, kernel_size=3, activation=True, norm=False, bias=True)
        self.latent_linear = EqualLinear(reducer_output_dim * reducer_output_nf,
                                         latent_bank_latent_dim * latent_bank_blocks,
                                         activation="fused_lrelu")
        self.latent_bank_blocks = latent_bank_blocks

    def forward(self, x):
        fea = self.initial_conv(x)
        fea = sequential_checkpoint(self.rrdb_blocks, len(self.rrdb_blocks), fea)
        rrdb_fea = fea
        convolutional_features = []
        for reducer in self.reducers:
            fea, f = checkpoint(reducer, fea)
            convolutional_features.append(f)

        latents = self.latent_conv(fea)
        latents = self.latent_linear(latents.flatten(1, -1)).view(fea.shape[0], self.latent_bank_blocks, -1)

        return rrdb_fea, convolutional_features, latents


# Produces an image by fusing the output features from the latent bank.
class GleanDecoder(nn.Module):
    # To determine latent_bank_filters, use the `self.channels` map for the desired input dimensions from stylegan2_rosinality.py
    def __init__(self, nf, latent_bank_filters=[512, 256, 128]):
        super().__init__()
        self.initial_conv = ConvGnLelu(nf, nf, kernel_size=3, activation=True, norm=False, bias=True, weight_init_factor=.1)

        decoder_block_shuffled_dims = [nf] + latent_bank_filters
        self.decoder_blocks = nn.ModuleList([ConvGnLelu(decoder_block_shuffled_dims[i] + latent_bank_filters[i],
                                                        latent_bank_filters[i],
                                                        kernel_size=3, bias=True, norm=False, activation=True,
                                                        weight_init_factor=.1)
                                             for i in range(len(latent_bank_filters))])

        final_dim = latent_bank_filters[-1]
        self.final_decode = ConvGnLelu(final_dim, 3, kernel_size=3, activation=False, bias=True, norm=False, weight_init_factor=.1)

    def forward(self, rrdb_fea, latent_bank_fea):
        fea = self.initial_conv(rrdb_fea)
        for i, block in enumerate(self.decoder_blocks):
            # The paper calls for PixelShuffle here, but I don't have good experience with that. It also doesn't align with the way the underlying StyleGAN works.
            fea = nn.functional.interpolate(fea, scale_factor=2, mode="nearest")
            fea = torch.cat([fea, latent_bank_fea[i]], dim=1)
            fea = checkpoint(block, fea)
        return self.final_decode(fea)


class GleanGenerator(nn.Module):
    def __init__(self, nf, latent_bank_pretrained_weights, latent_bank_max_dim=1024, gen_output_dim=256,
                 encoder_rrdb_nb=6, latent_bank_latent_dim=512, input_dim=32, initial_stride=1):
        super().__init__()
        self.input_dim = input_dim
        after_stride_dim = input_dim // initial_stride
        latent_blocks = int(math.log(gen_output_dim, 2))   # From 4x4->gen_output_dim x gen_output_dim + initial styled conv
        encoder_reductions = int(math.log(after_stride_dim / 4, 2)) + 1
        self.encoder = GleanEncoder(nf, encoder_rrdb_nb, reductions=encoder_reductions, latent_bank_blocks=latent_blocks,
                                    latent_bank_latent_dim=latent_bank_latent_dim, input_dim=after_stride_dim, initial_stride=initial_stride)
        decoder_blocks = int(math.log(gen_output_dim/after_stride_dim, 2))
        latent_bank_filters_out = [512, 512, 512, 256, 128]
        latent_bank_filters_out = latent_bank_filters_out[-decoder_blocks:]
        self.latent_bank = Stylegan2LatentBank(latent_bank_pretrained_weights, encoder_nf=nf, max_dim=latent_bank_max_dim,
                                               latent_dim=latent_bank_latent_dim, encoder_levels=encoder_reductions,
                                               decoder_levels=decoder_blocks)
        self.decoder = GleanDecoder(nf, latent_bank_filters_out)

    def forward(self, x):
        assert self.input_dim == x.shape[-1] and self.input_dim == x.shape[-2]
        rrdb_fea, conv_fea, latents = self.encoder(x)
        latent_bank_fea = self.latent_bank(conv_fea, latents)
        return self.decoder(rrdb_fea, latent_bank_fea)


@register_model
def register_glean(opt_net, opt):
    kwargs = {}
    allowlist = ['nf', 'latent_bank_pretrained_weights', 'latent_bank_max_dim', 'gen_output_dim', 'encoder_rrdb_nb', 'latent_bank_latent_dim',
                 'input_dim', 'initial_stride']
    for k, v in opt_net.items():
        if k in allowlist:
            kwargs[k] = v
    return GleanGenerator(**kwargs)
