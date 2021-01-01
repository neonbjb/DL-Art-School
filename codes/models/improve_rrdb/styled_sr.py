from math import log2
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.RRDBNet_arch import RRDB
from models.arch_util import ConvGnLelu, default_init_weights
from models.stylegan.stylegan2_lucidrains import StyleVectorizer, GeneratorBlock, Conv2DMod, leaky_relu, Blur
from trainer.networks import register_model
from utils.util import checkpoint


class EncoderRRDB(nn.Module):
    def __init__(self, mid_channels=64, output_channels=32, growth_channels=32, init_weight=.1):
        super(EncoderRRDB, self).__init__()
        for i in range(5):
            out_channels = output_channels if i == 4 else growth_channels
            self.add_module(
                f'conv{i+1}',
                nn.Conv2d(mid_channels + i * growth_channels, out_channels, 3,
                          1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        for i in range(5):
            default_init_weights(getattr(self, f'conv{i+1}'), init_weight)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


class StyledSrEncoder(nn.Module):
    def __init__(self, fea_out=256):
        super().__init__()
        # Current assumes fea_out=256.
        self.initial_conv = ConvGnLelu(3, 32, kernel_size=7, norm=False, activation=False, bias=True)
        self.rrdbs = nn.ModuleList([
           EncoderRRDB(32),
           EncoderRRDB(64),
           EncoderRRDB(96),
           EncoderRRDB(128),
           EncoderRRDB(160),
           EncoderRRDB(192),
           EncoderRRDB(224)])

    def forward(self, x):
        fea = self.initial_conv(x)
        for rrdb in self.rrdbs:
            fea = torch.cat([fea, checkpoint(rrdb, fea)], dim=1)
        return fea


class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, transparent=False, start_level=3, upsample_levels=2):
        super().__init__()
        total_levels = upsample_levels + 1  # The first level handles the raw encoder output and doesn't upsample.
        self.image_size = image_size
        self.scale = 2 ** upsample_levels
        self.latent_dim = latent_dim
        self.num_layers = total_levels
        filters = [
            512,  # 4x4
            512,  # 8x8
            512,  # 16x16
            256,  # 32x32
            128,  # 64x64
            64,   # 128x128
            32,   # 256x256
            16,   # 512x512
            8,    # 1024x1024
        ]

        self.encoder = StyledSrEncoder(filters[start_level])

        in_out_pairs = list(zip(filters[:-1], filters[1:]))
        self.blocks = nn.ModuleList([])
        for ind in range(start_level, start_level+total_levels):
            in_chan, out_chan = in_out_pairs[ind]
            not_first = ind != start_level
            not_last = ind != (start_level+total_levels-1)
            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
                rgba=transparent
            )
            self.blocks.append(block)

    def forward(self, lr, styles):
        b, c, h, w = lr.shape
        input_noise = torch.rand(b, h * self.scale, w * self.scale, 1).to(lr.device)

        rgb = lr
        styles = styles.transpose(0, 1)

        x = self.encoder(lr)
        for style, block in zip(styles, self.blocks):
            x, rgb = checkpoint(block, x, rgb, style, input_noise)

        return rgb


class StyledSrGenerator(nn.Module):
    def __init__(self, image_size, latent_dim=512, style_depth=8, lr_mlp=.1):
        super().__init__()
        self.vectorizer = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
        self.gen = Generator(image_size=image_size, latent_dim=latent_dim)
        self.mixed_prob = .9
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear} and hasattr(m, 'weight'):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.gen.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def forward(self, x):
        b, f, h, w = x.shape

        # Synthesize style latents from noise.
        style = torch.randn(b*2, self.gen.latent_dim).to(x.device)
        w = self.vectorizer(style)

        # Randomly distribute styles across layers
        w_styles = w[:,None,:].expand(-1, self.gen.num_layers, -1).clone()
        for j in range(b):
            cutoff = int(torch.rand(()).numpy() * self.gen.num_layers)
            if cutoff == self.gen.num_layers or random() > self.mixed_prob:
                w_styles[j] = w_styles[j*2]
            else:
                w_styles[j, :cutoff] = w_styles[j*2, :cutoff]
                w_styles[j, cutoff:] = w_styles[j*2+1, cutoff:]
        w_styles = w_styles[:b]

        out = self.gen(x, w_styles)

        # Compute the net, areal, pixel-wise additions made on top of the LR image.
        out_down = F.interpolate(out, size=(x.shape[-2], x.shape[-1]), mode="area")
        diff = torch.sum(torch.abs(out_down - x), dim=[1,2,3])

        return out, diff, w_styles


if __name__ == '__main__':
    gen = StyledSrGenerator(128)
    out = gen(torch.rand(1,3,32,32))
    print([o.shape for o in out])


@register_model
def register_opt_styled_sr(opt_net, opt):
    return StyledSrGenerator(128)
