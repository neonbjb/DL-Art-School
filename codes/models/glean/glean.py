import math

import torch.nn as nn
import torch

from models.RRDBNet_arch import RRDB
from models.arch_util import ConvGnLelu


# Produces a convolutional feature (`f`) and a reduced feature map with double the filters.
from models.glean.stylegan2_latent_bank import Stylegan2LatentBank
from models.stylegan.stylegan2_rosinality import EqualLinear
from utils.util import checkpoint, sequential_checkpoint


class GleanEncoderBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.structural_latent_conv = ConvGnLelu(nf, nf, kernel_size=1, activation=False, norm=False, bias=True)
        self.process = nn.Sequential(
            ConvGnLelu(nf, nf*2, kernel_size=3, stride=2, activation=True, norm=False, bias=False),
            ConvGnLelu(nf*2, nf*2, kernel_size=3, activation=True, norm=False, bias=False)
        )

    def forward(self, x):
        structural_latent = self.structural_latent_conv(x)
        fea = self.process(x)
        return fea, structural_latent


# Produces RRDB features, a list of convolutional features (`f` shape=[l][b,c,h,w] l=levels aka f_sub)
# and latent vectors (`C` shape=[b,l,f] l=levels aka C_sub) for use with the latent bank.
# Note that latent levels and convolutional feature levels do not necessarily match, per the paper.
class GleanEncoder(nn.Module):
    def __init__(self, nf, nb, reductions=4, latent_bank_blocks=13, latent_bank_latent_dim=512, input_dim=32):
        super().__init__()
        self.initial_conv = ConvGnLelu(3, nf, kernel_size=7, activation=False, norm=False, bias=True)
        self.rrdb_blocks = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        self.reducers = nn.ModuleList([GleanEncoderBlock(nf * 2 ** i) for i in range(reductions)])

        reducer_output_dim = (input_dim // (2 ** reductions)) ** 2
        reducer_output_nf = nf * 2 ** reductions
        self.latent_conv = ConvGnLelu(reducer_output_nf, reducer_output_nf, kernel_size=1, activation=True, norm=False, bias=True)
        # This is a questionable part of this architecture. Apply multiple Denses to separate outputs (as I've done here)?
        # Apply a single dense, then split the outputs? Who knows..
        self.latent_linears = nn.ModuleList([EqualLinear(reducer_output_dim * reducer_output_nf, latent_bank_latent_dim,
                                                        activation="fused_lrelu")
                                            for _ in range(latent_bank_blocks)])

    def forward(self, x):
        fea = self.initial_conv(x)
        fea = sequential_checkpoint(self.rrdb_blocks, len(self.rrdb_blocks), fea)
        rrdb_fea = fea
        convolutional_features = []
        for reducer in self.reducers:
            fea, f = checkpoint(reducer, fea)
            convolutional_features.append(f)

        latents = self.latent_conv(fea)
        latents = [dense(latents.flatten(1, -1)) for dense in self.latent_linears]
        latents = torch.stack(latents, dim=1)

        return rrdb_fea, convolutional_features, latents


# Produces an image by fusing the output features from the latent bank.
class GleanDecoder(nn.Module):
    # To determine latent_bank_filters, use the `self.channels` map for the desired input dimensions from stylegan2_rosinality.py
    def __init__(self, nf, latent_bank_filters=[512, 256, 128]):
        super().__init__()
        self.initial_conv = ConvGnLelu(nf, nf, kernel_size=3, activation=False, norm=False, bias=True)

        # The paper calls for pixel shuffling each output of the decoder. We need to make sure that is possible. Doing it by using the latent bank filters as the output filters for each decoder stage
        assert latent_bank_filters[-1] % 4 == 0
        decoder_block_shuffled_dims = [nf // 4]
        decoder_block_shuffled_dims.extend([l // 4 for l in latent_bank_filters])
        self.decoder_blocks = nn.ModuleList([ConvGnLelu(decoder_block_shuffled_dims[i] + latent_bank_filters[i],
                                                        latent_bank_filters[i],
                                                        kernel_size=3, bias=True, norm=False, activation=False)
                                             for i in range(len(latent_bank_filters))])
        self.shuffler = nn.PixelShuffle(2)  # TODO: I'm a bit skeptical about this. It doesn't align with RRDB or StyleGAN. It also always produces artifacts in my experience. Try using interpolation instead.

        final_dim = latent_bank_filters[-1]
        self.final_decode = nn.Sequential(ConvGnLelu(final_dim, final_dim, kernel_size=3, activation=True, bias=True, norm=False),
                                          ConvGnLelu(final_dim, 3, kernel_size=3, activation=False, bias=True, norm=False))

    def forward(self, rrdb_fea, latent_bank_fea):
        fea = self.initial_conv(rrdb_fea)
        for i, block in enumerate(self.decoder_blocks):
            fea = self.shuffler(fea)
            fea = torch.cat([fea, latent_bank_fea[i]], dim=1)
            fea = checkpoint(block, fea)
        return self.final_decode(fea)


class GleanGenerator(nn.Module):
    def __init__(self, nf, latent_bank_pretrained_weights, latent_bank_max_dim=1024, gen_output_dim=256,
                 encoder_rrdb_nb=6, encoder_reductions=4, latent_bank_latent_dim=512, input_dim=32):
        super().__init__()
        self.input_dim = input_dim
        latent_blocks = int(math.log(gen_output_dim, 2)) - 1  # From 4x4->gen_output_dim x gen_output_dim
        latent_blocks = latent_blocks * 2 + 1  # Two styled convolutions per block, + an initial styled conv.
        self.encoder = GleanEncoder(nf, encoder_rrdb_nb, reductions=encoder_reductions, latent_bank_blocks=latent_blocks * 2 + 1,
                                    latent_bank_latent_dim=latent_bank_latent_dim, input_dim=input_dim)
        decoder_blocks = int(math.log(gen_output_dim/input_dim, 2))
        latent_bank_filters_out = [512, 256, 128]  # TODO: Use decoder_blocks to synthesize the correct value for latent_bank_filters here. The fixed defaults will work fine for testing, though.
        self.latent_bank = Stylegan2LatentBank(latent_bank_pretrained_weights, encoder_nf=nf, max_dim=latent_bank_max_dim,
                                               latent_dim=latent_bank_latent_dim, encoder_levels=encoder_reductions,
                                               decoder_levels=decoder_blocks)
        self.decoder = GleanDecoder(nf, latent_bank_filters_out)

    def forward(self, x):
        assert self.input_dim == x.shape[-1] and self.input_dim == x.shape[-2]
        rrdb_fea, conv_fea, latents = self.encoder(x)
        latent_bank_fea = self.latent_bank(conv_fea, latents)
        return self.decoder(rrdb_fea, latent_bank_fea)
