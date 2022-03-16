import torch
import torch.nn as nn

from models.arch_util import ConvGnLelu
from models.image_generation.stylegan.stylegan2_rosinality import Generator


class Stylegan2LatentBank(nn.Module):
    def __init__(self, pretrained_model_file, encoder_nf=64, encoder_max_nf=512, max_dim=1024, latent_dim=512, encoder_levels=4, decoder_levels=3):
        super().__init__()

        # Initialize the bank.
        self.bank = Generator(size=max_dim, style_dim=latent_dim, n_mlp=8, channel_multiplier=2)  # Assumed using 'f' generators with mult=2.
        state_dict = torch.load(pretrained_model_file)
        self.bank.load_state_dict(state_dict, strict=True)

        # Shut off training of the latent bank.
        for p in self.bank.parameters():
            p.requires_grad = False
            p.DO_NOT_TRAIN = True

        # These are from `stylegan_rosinality.py`, search for `self.channels = {`.
        stylegan_encoder_dims = [512, 512, 512, 512, 512, 256, 128, 64, 32]

        # Initialize the fusion blocks. TODO: Try using the StyledConvs instead of regular ones.
        encoder_output_dims = reversed([min(encoder_nf * 2 ** i, encoder_max_nf) for i in range(encoder_levels)])
        input_dims_by_layer = [eod + sed for eod, sed in zip(encoder_output_dims, stylegan_encoder_dims)]
        self.fusion_blocks = nn.ModuleList([ConvGnLelu(in_filters, out_filters, kernel_size=3, activation=True, norm=False, bias=True)
                                            for in_filters, out_filters in zip(input_dims_by_layer, stylegan_encoder_dims)])

        self.decoder_levels = decoder_levels
        self.decoder_start = encoder_levels - 1
        self.total_levels = encoder_levels + decoder_levels - 1

    # This forward mirrors the forward() pass from the rosinality stylegan2 implementation, with the additions called
    # for from the GLEAN paper. GLEAN mods are annotated with comments.
    # Removed stuff:
    # - Support for split latents (we're spoonfeeding them)
    # - Support for fixed noise inputs
    # - RGB computations -> we only care about the latents
    # - Style MLP -> GLEAN computes the Style inputs directly.
    # - Later layers -> GLEAN terminates at 256 resolution.
    def forward(self, convolutional_features, latent_vectors):

        out = self.bank.input(latent_vectors[:, 0])  # The input here is only used to fetch the batch size.
        out = self.bank.conv1(out, latent_vectors[:, 0], noise=None)

        k = 0
        decoder_outputs = []
        for conv1, conv2 in zip(self.bank.convs[::2], self.bank.convs[1::2]):
            if k < len(self.fusion_blocks):
                out = torch.cat([convolutional_features[-k-1], out], dim=1)
                out = self.fusion_blocks[k](out)

            out = conv1(out, latent_vectors[:, k], noise=None)
            out = conv2(out, latent_vectors[:, k], noise=None)

            if k >= self.decoder_start:
                decoder_outputs.append(out)
            if k >= self.total_levels:
                break

            k += 1

        return decoder_outputs
