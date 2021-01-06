from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.arch_util import kaiming_init
from models.styled_sr.stylegan2_base import StyleVectorizer, GeneratorBlock
from models.styled_sr.transfer_primitives import TransferConvGnLelu, TransferConv2d, TransferLinear
from trainer.networks import register_model
from utils.util import checkpoint, opt_get


def rrdb_init_weights(module, scale=1):
    for m in module.modules():
        if isinstance(m, TransferConv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, TransferLinear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale


class EncoderRRDB(nn.Module):
    def __init__(self, mid_channels=64, output_channels=32, growth_channels=32, init_weight=.1, transfer_mode=False):
        super(EncoderRRDB, self).__init__()
        for i in range(5):
            out_channels = output_channels if i == 4 else growth_channels
            self.add_module(
                f'conv{i+1}',
                TransferConv2d(mid_channels + i * growth_channels, out_channels, 3, 1, 1, transfer_mode=transfer_mode))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        for i in range(5):
            rrdb_init_weights(getattr(self, f'conv{i+1}'), init_weight)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


class StyledSrEncoder(nn.Module):
    def __init__(self, fea_out=256, initial_stride=1, transfer_mode=False):
        super().__init__()
        # Current assumes fea_out=256.
        self.initial_conv = TransferConvGnLelu(3, 32, kernel_size=7, stride=initial_stride, norm=False, activation=False, bias=True, transfer_mode=transfer_mode)
        self.rrdbs = nn.ModuleList([
           EncoderRRDB(32, transfer_mode=transfer_mode),
           EncoderRRDB(64, transfer_mode=transfer_mode),
           EncoderRRDB(96, transfer_mode=transfer_mode),
           EncoderRRDB(128, transfer_mode=transfer_mode),
           EncoderRRDB(160, transfer_mode=transfer_mode),
           EncoderRRDB(192, transfer_mode=transfer_mode),
           EncoderRRDB(224, transfer_mode=transfer_mode)])

    def forward(self, x):
        fea = self.initial_conv(x)
        for rrdb in self.rrdbs:
            fea = torch.cat([fea, checkpoint(rrdb, fea)], dim=1)
        return fea


class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, initial_stride=1, start_level=3, upsample_levels=2, transfer_mode=False):
        super().__init__()
        total_levels = upsample_levels + 1  # The first level handles the raw encoder output and doesn't upsample.
        self.image_size = image_size
        self.scale = 2 ** upsample_levels
        self.latent_dim = latent_dim
        self.num_layers = total_levels
        self.transfer_mode = transfer_mode
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

        # I'm making a guess here that the encoder does not need transfer learning, hence fixed transfer_mode=False. This should be vetted.
        self.encoder = StyledSrEncoder(filters[start_level], initial_stride, transfer_mode=False)

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
                transfer_learning_mode=transfer_mode
            )
            self.blocks.append(block)

    def forward(self, lr, styles):
        b, c, h, w = lr.shape
        if self.transfer_mode:
            with torch.no_grad():
                x = self.encoder(lr)
        else:
            x = self.encoder(lr)

        styles = styles.transpose(0, 1)
        input_noise = torch.rand(b, h * self.scale, w * self.scale, 1).to(lr.device)
        if h != x.shape[-2]:
            rgb = F.interpolate(lr, size=x.shape[2:], mode="area")
        else:
            rgb = lr

        for style, block in zip(styles, self.blocks):
            x, rgb = checkpoint(block, x, rgb, style, input_noise)

        return rgb


class StyledSrGenerator(nn.Module):
    def __init__(self, image_size, initial_stride=1, latent_dim=512, style_depth=8, lr_mlp=.1, transfer_mode=False):
        super().__init__()
        # Assume the vectorizer doesnt need transfer_mode=True. Re-evaluate this later.
        self.vectorizer = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp, transfer_mode=False)
        self.gen = Generator(image_size=image_size, latent_dim=latent_dim, initial_stride=initial_stride, transfer_mode=transfer_mode)
        self.l2 = nn.MSELoss()
        self.mixed_prob = .9
        self._init_weights()
        self.transfer_mode = transfer_mode
        self.initial_stride = initial_stride
        if transfer_mode:
            for p in self.parameters():
                if not hasattr(p, 'FOR_TRANSFER_LEARNING'):
                    p.DO_NOT_TRAIN = True


    def _init_weights(self):
        for m in self.modules():
            if type(m) in {TransferConv2d, TransferLinear} and hasattr(m, 'weight'):
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
        if self.transfer_mode:
            with torch.no_grad():
                w = self.vectorizer(style)
        else:
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

        # Compute an L2 loss on the areal interpolation of the generated image back down to LR * initial_stride; used
        # for regularization.
        out_down = F.interpolate(out, size=(x.shape[-2] // self.initial_stride, x.shape[-1] // self.initial_stride), mode="area")
        if self.initial_stride > 1:
            x = F.interpolate(x, scale_factor=1/self.initial_stride, mode="area")
        l2_reg = self.l2(x, out_down)

        return out, l2_reg, w_styles


if __name__ == '__main__':
    gen = StyledSrGenerator(128, 2)
    out = gen(torch.rand(1,3,64,64))
    print([o.shape for o in out])


@register_model
def register_styled_sr(opt_net, opt):
    return StyledSrGenerator(128,
                             initial_stride=opt_get(opt_net, ['initial_stride'], 1),
                             transfer_mode=opt_get(opt_net, ['transfer_mode'], False))
