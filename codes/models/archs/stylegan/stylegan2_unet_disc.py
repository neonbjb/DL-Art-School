from functools import partial
from math import log2
from random import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.archs.stylegan.stylegan2 import attn_and_ff
from models.steps.losses import ConfigurableLoss


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p)


def double_conv(chan_in, chan_out):
    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, 3, padding=1),
        leaky_relu(),
        nn.Conv2d(chan_out, chan_out, 3, padding=1),
        leaky_relu()
    )


class Flatten(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    def forward(self, x):
        return x.flatten(self.index)


class DownBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = double_conv(input_channels, filters)
        self.down = nn.Conv2d(filters, filters, 3, padding = 1, stride = 2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        unet_res = x

        if self.down is not None:
            x = self.down(x)

        x = x + res
        return x, unet_res


class UpBlock(nn.Module):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.conv_res = nn.ConvTranspose2d(input_channels // 2, filters, 1, stride = 2)
        self.net = double_conv(input_channels, filters)
        self.up = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False)
        self.input_channels = input_channels
        self.filters = filters

    def forward(self, x, res):
        *_, h, w = x.shape
        conv_res = self.conv_res(x, output_size = (h * 2, w * 2))
        x = self.up(x)
        x = torch.cat((x, res), dim=1)
        x = self.net(x)
        x = x + conv_res
        return x


class StyleGan2UnetDiscriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16, fmap_max = 512, input_filters=3):
        super().__init__()
        num_layers = int(log2(image_size) - 3)

        blocks = []
        filters = [input_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        filters[-1] = filters[-2]

        chan_in_out = list(zip(filters[:-1], filters[1:]))
        chan_in_out = list(map(list, chan_in_out))

        down_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DownBlock(in_chan, out_chan, downsample = is_not_last)
            down_blocks.append(block)

            attn_fn = attn_and_ff(out_chan)
            attn_blocks.append(attn_fn)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        last_chan = filters[-1]

        self.to_logit = nn.Sequential(
            leaky_relu(),
            nn.AvgPool2d(image_size // (2 ** num_layers)),
            Flatten(1),
            nn.Linear(last_chan, 1)
        )

        self.conv = double_conv(last_chan, last_chan)

        dec_chan_in_out = chan_in_out[:-1][::-1]
        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0]), dec_chan_in_out)))
        self.conv_out = nn.Conv2d(input_filters, 1, 1)

    def forward(self, x):
        b, *_ = x.shape

        residuals = []

        for (down_block, attn_block) in zip(self.down_blocks, self.attn_blocks):
            x, unet_res = down_block(x)
            residuals.append(unet_res)

            if attn_block is not None:
                x = attn_block(x)

        x = self.conv(x) + x
        enc_out = self.to_logit(x)

        for (up_block, res) in zip(self.up_blocks, residuals[:-1][::-1]):
            x = up_block(x, res)

        dec_out = self.conv_out(x)
        return dec_out, enc_out


def warmup(start, end, max_steps, current_step):
    if current_step > max_steps:
        return end
    return (end - start) * (current_step / max_steps) + start


def mask_src_tgt(source, target, mask):
    return source * mask + (1 - mask) * target


def cutmix(source, target, coors, alpha = 1.):
    source, target = map(torch.clone, (source, target))
    ((y0, y1), (x0, x1)), _ = coors
    source[:, :, y0:y1, x0:x1] = target[:, :, y0:y1, x0:x1]
    return source


def cutmix_coordinates(height, width, alpha = 1.):
    lam = np.random.beta(alpha, alpha)

    cx = np.random.uniform(0, width)
    cy = np.random.uniform(0, height)
    w = width * np.sqrt(1 - lam)
    h = height * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, width)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, height)))

    return ((y0, y1), (x0, x1)), lam


class StyleGan2UnetDivergenceLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.real = opt['real']
        self.fake = opt['fake']
        self.discriminator = opt['discriminator']
        self.for_gen = opt['gen_loss']
        self.gp_frequency = opt['gradient_penalty_frequency']
        self.noise = opt['noise'] if 'noise' in opt.keys() else 0
        self.image_size = opt['image_size']

    def forward(self, net, state):
        real_input = state[self.real]
        fake_input = state[self.fake]
        if self.noise != 0:
            fake_input = fake_input + torch.rand_like(fake_input) * self.noise
            real_input = real_input + torch.rand_like(real_input) * self.noise

        D = self.env['discriminators'][self.discriminator]
        fake_dec, fake_enc = D(fake_input)
        fake_aug_images = D.aug_images
        if self.for_gen:
            return fake_enc.mean() + F.relu(1 + fake_dec).mean()
        else:
            dec_loss_coef = warmup(0, 1., 30000, self.env['step'])
            cutmix_prob = warmup(0, 0.25, 30000, self.env['step'])
            apply_cutmix = random() < cutmix_prob

            real_input.requires_grad_()  # <-- Needed to compute gradients on the input.
            real_dec, real_enc = D(real_input)
            real_aug_images = D.aug_images
            enc_divergence = (F.relu(1 + real_enc) + F.relu(1 - fake_enc)).mean()
            dec_divergence = (F.relu(1 + real_dec) + F.relu(1 - fake_dec)).mean()
            divergence_loss = enc_divergence + dec_divergence * dec_loss_coef

            if apply_cutmix:
                mask = cutmix(
                    torch.ones_like(real_dec),
                    torch.zeros_like(real_dec),
                    cutmix_coordinates(self.image_size, self.image_size)
                )

                if random() > 0.5:
                    mask = 1 - mask

                cutmix_images = mask_src_tgt(real_aug_images, fake_aug_images, mask)
                cutmix_enc_out, cutmix_dec_out = self.GAN.D(cutmix_images)

                cutmix_enc_divergence = F.relu(1 - cutmix_enc_out).mean()
                cutmix_dec_divergence = F.relu(1 + (mask * 2 - 1) * cutmix_dec_out).mean()
                disc_loss = divergence_loss + cutmix_enc_divergence + cutmix_dec_divergence

                cr_cutmix_dec_out = mask_src_tgt(real_dec, fake_dec, mask)
                cr_loss = F.mse_loss(cutmix_dec_out, cr_cutmix_dec_out) * self.cr_weight
                self.last_cr_loss = cr_loss.clone().detach().item()

                disc_loss = disc_loss + cr_loss * dec_loss_coef

            # Apply gradient penalty. TODO: migrate this elsewhere.
            if self.env['step'] % self.gp_frequency == 0:
                from models.archs.stylegan.stylegan2 import gradient_penalty
                gp = gradient_penalty(real_input, real)
                self.metrics.append(("gradient_penalty", gp.clone().detach()))
                disc_loss = disc_loss + gp

            real_input.requires_grad_(requires_grad=False)
            return disc_loss