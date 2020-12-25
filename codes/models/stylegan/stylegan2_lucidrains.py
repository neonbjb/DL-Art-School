import functools
import math
import multiprocessing
from contextlib import contextmanager, ExitStack
from functools import partial
from math import log2
from random import random

import torch
import torch.nn.functional as F
import trainer.losses as L
import numpy as np

from kornia.filters import filter2D
from linear_attention_transformer import ImageLinearAttention
from torch import nn
from torch.autograd import grad as torch_grad
from vector_quantize_pytorch import VectorQuantize

from trainer.networks import register_model
from utils.util import checkpoint

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

num_cores = multiprocessing.cpu_count()

# constants

EPS = 1e-8
CALC_FID_NUM_IMAGES = 12800


# helper classes

def DiffAugment(x, types=[]):
    for p in types:
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x.contiguous()

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

class NanException(Exception):
    pass


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2D(x, f, normalized=True)


# one layer of self-attention and feedforward, for images

attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan, norm_queries=True))),
    Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])


# helpers

def exists(val):
    return val is not None


@contextmanager
def null_context():
    yield


def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]

    return multi_contexts


def default(value, d):
    return value if exists(value) else d


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def cast_list(el):
    return el if isinstance(el, list) else [el]


def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return not exists(t)


def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException


def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts = head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield


def loss_backwards(fp16, loss, optimizer, loss_id, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


def gradient_penalty(images, output, weight=10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def calc_pl_lengths(styles, images):
    device = images.device
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape, device=device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()


def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda(device)


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

# augmentations

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))


class StyleGan2Augmentor(nn.Module):
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


# stylegan2 classes

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul=0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba=False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        from models.archs.arch_util import ConvGnLelu
        self.style2scale = ConvGnLelu(style_dim, in_channel, kernel_size=1, norm=False, activation=False, bias=True)
        self.style2bias = ConvGnLelu(style_dim, in_channel, kernel_size=1, norm=False, activation=False, bias=True, weight_init_factor=0)
        self.norm = nn.InstanceNorm2d(in_channel)

    def forward(self, input, style):
        gamma = self.style2scale(style)
        beta = self.style2bias(style)
        out = self.norm(input)
        out = gamma * out + beta
        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + EPS)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x


class GeneratorBlockWithStructure(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample=True, upsample_rgb=True, rgba=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        # Uses stylegan1 style blocks for injecting structural latent.
        self.conv0 = EqualConv2d(input_channels, filters, 3, padding=1)
        self.to_noise0 = nn.Linear(1, filters)
        self.noise0 = equal_lr(NoiseInjection(filters))
        self.adain0 = AdaptiveInstanceNorm(filters, latent_dim)

        self.to_style1 = nn.Linear(latent_dim, filters)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(filters, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise, structure_input):
        if exists(self.upsample):
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise0 = self.to_noise0(inoise).permute((0, 3, 1, 2))
        noise1 = self.to_noise1(inoise).permute((0, 3, 1, 2))
        noise2 = self.to_noise2(inoise).permute((0, 3, 1, 2))

        structure = torch.nn.functional.interpolate(structure_input, size=x.shape[2:], mode="nearest")
        x = self.conv0(x)
        x = self.noise0(x, noise0)
        x = self.adain0(x, structure)

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb


class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample=True, upsample_rgb=True, rgba=False, structure_input=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.structure_input = structure_input
        if self.structure_input:
            self.structure_conv = nn.Conv2d(3, input_channels, 3, padding=1)
            input_channels = input_channels * 2

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise, structure_input=None):
        if exists(self.upsample):
            x = self.upsample(x)

        if self.structure_input:
            s = self.structure_conv(structure_input)
            x = torch.cat([x, s], dim=1)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb


class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity=16, transparent=False, attn_layers=[], no_const=False,
                 fmap_max=512, structure_input=False):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            if structure_input:
                block_fn = GeneratorBlockWithStructure
            else:
                block_fn = GeneratorBlock

            block = block_fn(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
                rgba=transparent
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise, structure_input=None, starting_shape=None):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)
        if starting_shape is not None:
            x = F.interpolate(x, size=starting_shape, mode="bilinear")

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        if structure_input is not None:
            s = torch.nn.functional.interpolate(structure_input, size=x.shape[2:], mode="nearest")
        for style, block, attn in zip(styles, self.blocks, self.attns):
            if exists(attn):
                x = checkpoint(attn, x)
            if structure_input is not None:
                if exists(block.upsample):
                    # In this case, the structural guidance is given by the extra information over the previous layer.
                    twoX = (x.shape[2]*2, x.shape[3]*2)
                    sn = torch.nn.functional.interpolate(structure_input, size=twoX, mode="nearest")
                    s_int = torch.nn.functional.interpolate(s, size=twoX, mode="bilinear")
                    s_diff = sn - s_int
                else:
                    # This is the initial case - just feed in the base structure.
                    s_diff = s
            else:
                s_diff = None
            x, rgb = checkpoint(block, x, rgb, style, input_noise, s_diff)

        return rgb


# Wrapper that combines style vectorizer with the actual generator.
class StyleGan2GeneratorWithLatent(nn.Module):
    def __init__(self, image_size, latent_dim=512, style_depth=8, lr_mlp=.1, network_capacity=16, transparent=False,
                 attn_layers=[], no_const=False, fmap_max=512, structure_input=False):
        super().__init__()
        self.vectorizer = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
        self.gen = Generator(image_size, latent_dim, network_capacity, transparent, attn_layers, no_const, fmap_max,
                             structure_input=structure_input)
        self.mixed_prob = .9
        self._init_weights()

    def noise(self, n, latent_dim, device):
        return torch.randn(n, latent_dim).cuda(device)

    def noise_list(self, n, layers, latent_dim, device):
        return [(self.noise(n, latent_dim, device), layers)]

    def mixed_list(self, n, layers, latent_dim, device):
        tt = int(torch.rand(()).numpy() * layers)
        return self.noise_list(n, tt, latent_dim, device) + self.noise_list(n, layers - tt, latent_dim, device)

    def latent_to_w(self, style_vectorizer, latent_descr):
        return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

    def styles_def_to_tensor(self, styles_def):
        return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

    # To use per the stylegan paper, input should be uniform noise. This gen takes it in as a normal "image" format:
    # b,f,h,w.
    def forward(self, x, structure_input=None, fit_starting_shape_to_structure=False):
        b, f, h, w = x.shape

        full_random_latents = True
        if full_random_latents:
            style = self.noise(b*2, self.gen.latent_dim, x.device)
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
        else:
            get_latents_fn = self.mixed_list if random() < self.mixed_prob else self.noise_list
            style = get_latents_fn(b, self.gen.num_layers, self.gen.latent_dim, device=x.device)
            w_space = self.latent_to_w(self.vectorizer, style)
            w_styles = self.styles_def_to_tensor(w_space)

        starting_shape = None
        if fit_starting_shape_to_structure:
            starting_shape = (x.shape[2] // 32, x.shape[3] // 32)
        # The underlying model expects the noise as b,h,w,1. Make it so.
        return self.gen(w_styles, x[:,0,:,:].unsqueeze(dim=3), structure_input, starting_shape), w_styles

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear} and hasattr(m, 'weight'):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.gen.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
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
                 transparent=False, fmap_max=512, input_filters=3):
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

            quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

        self._init_weights()

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
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


class StyleGan2DivergenceLoss(L.ConfigurableLoss):
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
                from models.archs.stylegan.stylegan2_lucidrains import gradient_penalty
                gp = gradient_penalty(real_input, real)
                self.metrics.append(("gradient_penalty", gp.clone().detach()))
                divergence_loss = divergence_loss + gp

            real_input.requires_grad_(requires_grad=False)
            return divergence_loss


class StyleGan2PathLengthLoss(L.ConfigurableLoss):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.w_styles = opt['w_styles']
        self.gen = opt['gen']
        self.pl_mean = None
        from models.archs.stylegan.stylegan2_lucidrains import EMA
        self.pl_length_ma = EMA(.99)

    def forward(self, net, state):
        w_styles = state[self.w_styles]
        gen = state[self.gen]
        from models.archs.stylegan.stylegan2_lucidrains import calc_pl_lengths
        pl_lengths = calc_pl_lengths(w_styles, gen)
        avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

        from models.archs.stylegan.stylegan2_lucidrains import is_empty
        if not is_empty(self.pl_mean):
            pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
            if not torch.isnan(pl_loss):
                return pl_loss
            else:
                print("Path length loss returned NaN!")

        self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
        return 0


@register_model
def register_stylegan2_lucidrains(opt_net, opt):
    is_structured = opt_net['structured'] if 'structured' in opt_net.keys() else False
    attn = opt_net['attn_layers'] if 'attn_layers' in opt_net.keys() else []
    return StyleGan2GeneratorWithLatent(image_size=opt_net['image_size'], latent_dim=opt_net['latent_dim'],
                                        style_depth=opt_net['style_depth'], structure_input=is_structured,
                                        attn_layers=attn)
