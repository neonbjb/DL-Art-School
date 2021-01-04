# Heavily based on the lucidrains stylegan2 discriminator implementation.
from functools import partial
from math import log2
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import trainer.losses as L
from vector_quantize_pytorch import VectorQuantize

from models.styled_sr.stylegan2_base import attn_and_ff, PermuteToFrom, Blur, leaky_relu, exists
from trainer.networks import register_model
from utils.util import checkpoint, opt_get


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.filters = filters
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2d(filters, filters, 3, padding=1, stride=2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


class StyleGan2Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity=16, fq_layers=[], fq_dict_size=256, attn_layers=[],
                 transparent=False, fmap_max=512, input_filters=3, quantize=False, do_checkpointing=False, mlp=False):
        super().__init__()
        num_layers = int(log2(image_size) - 1)

        blocks = []
        filters = [input_filters] + [(64) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            if quantize:
                quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
                quantize_blocks.append(quantize_fn)
            else:
                quantize_blocks.append(None)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)
        self.do_checkpointing = do_checkpointing

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = nn.Flatten()
        if mlp:
            self.to_logit = nn.Sequential(nn.Linear(latent_dim, 100),
                                          leaky_relu(),
                                          nn.Linear(100, 1))
        else:
            self.to_logit = nn.Linear(latent_dim, 1)

        self._init_weights()

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            if self.do_checkpointing:
                x = checkpoint(block, x)
            else:
                x = block(x)

            if exists(attn_block):
                x = attn_block(x)

            if exists(q_block):
                x, _, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        if exists(q_block):
            return x.squeeze(), quantize_loss
        else:
            return x.squeeze()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    # Configures the network as partially pre-trained. This means:
    # 1) The top (high-resolution) `num_blocks` will have their weights re-initialized.
    # 2) The haed (linear layers) will also have their weights re-initialized
    # 3) All intermediate blocks will be frozen until step `frozen_until_step`
    # These settings will be applied after the weights have been loaded (network_loaded())
    def configure_partial_training(self, bypass_blocks=0, num_blocks=2, frozen_until_step=0):
        self.bypass_blocks = bypass_blocks
        self.num_blocks = num_blocks
        self.frozen_until_step = frozen_until_step

    # Called after the network weights are loaded.
    def network_loaded(self):
        if not hasattr(self, 'frozen_until_step'):
            return

        if self.bypass_blocks > 0:
            self.blocks = self.blocks[self.bypass_blocks:]
            self.blocks[0] = DiscriminatorBlock(3, self.blocks[0].filters, downsample=True).to(next(self.parameters()).device)

        reset_blocks = [self.to_logit]
        for i in range(self.num_blocks):
            reset_blocks.append(self.blocks[i])
        for bl in reset_blocks:
            for m in bl.modules():
                if type(m) in {nn.Conv2d, nn.Linear}:
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                for p in m.parameters(recurse=True):
                    p._NEW_BLOCK = True
        for p in self.parameters():
            if not hasattr(p, '_NEW_BLOCK'):
                p.DO_NOT_TRAIN_UNTIL = self.frozen_until_step


# helper classes
def DiffAugment(x, types=[]):
    for p in types:
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x.contiguous()


def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}


class DiscAugmentor(nn.Module):
    def __init__(self, D, image_size, types, prob):
        super().__init__()
        self.D = D
        self.prob = prob
        self.types = types

    def forward(self, images, detach=False):
        if random() < self.prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=self.types)

        if detach:
            images = images.detach()

        # Save away for use elsewhere (e.g. unet loss)
        self.aug_images = images

        return self.D(images)

    def network_loaded(self):
        self.D.network_loaded()


def loss_backwards(fp16, loss, optimizer, loss_id, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


def gradient_penalty(images, output, weight=10, return_structured_grads=False):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    flat_grad = gradients.reshape(batch_size, -1)
    penalty = weight * ((flat_grad.norm(2, dim=1) - 1) ** 2).mean()
    if return_structured_grads:
        return penalty, gradients
    else:
        return penalty


class StyleSrGanDivergenceLoss(L.ConfigurableLoss):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.real = opt['real']
        self.fake = opt['fake']
        self.discriminator = opt['discriminator']
        self.for_gen = opt['gen_loss']
        self.gp_frequency = opt['gradient_penalty_frequency']
        self.noise = opt['noise'] if 'noise' in opt.keys() else 0

    def forward(self, net, state):
        real_input = state[self.real]
        fake_input = state[self.fake]
        if self.noise != 0:
            fake_input = fake_input + torch.rand_like(fake_input) * self.noise
            real_input = real_input + torch.rand_like(real_input) * self.noise

        D = self.env['discriminators'][self.discriminator]
        fake = D(fake_input)
        if self.for_gen:
            return fake.mean()
        else:
            real_input.requires_grad_()  # <-- Needed to compute gradients on the input.
            real = D(real_input)
            divergence_loss = (F.relu(1 + real) + F.relu(1 - fake)).mean()

            # Apply gradient penalty. TODO: migrate this elsewhere.
            if self.env['step'] % self.gp_frequency == 0:
                gp = gradient_penalty(real_input, real)
                self.metrics.append(("gradient_penalty", gp.clone().detach()))
                divergence_loss = divergence_loss + gp

            real_input.requires_grad_(requires_grad=False)
            return divergence_loss


@register_model
def register_styledsr_discriminator(opt_net, opt):
    attn = opt_net['attn_layers'] if 'attn_layers' in opt_net.keys() else []
    disc = StyleGan2Discriminator(image_size=opt_net['image_size'], input_filters=opt_net['in_nc'], attn_layers=attn,
                                  do_checkpointing=opt_get(opt_net, ['do_checkpointing'], False),
                                  quantize=opt_get(opt_net, ['quantize'], False),
                                  mlp=opt_get(opt_net, ['mlp_head'], True))
    if 'use_partial_pretrained' in opt_net.keys():
        disc.configure_partial_training(opt_net['bypass_blocks'], opt_net['partial_training_blocks'], opt_net['intermediate_blocks_frozen_until'])
    return DiscAugmentor(disc, opt_net['image_size'], types=opt_net['augmentation_types'], prob=opt_net['augmentation_probability'])
