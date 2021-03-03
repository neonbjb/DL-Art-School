import math
import multiprocessing
import random
from contextlib import contextmanager, ExitStack
from functools import partial
from math import log2, floor
from pathlib import Path
from random import random

import torch
import torch.nn.functional as F
from gsa_pytorch import GSA

import trainer.losses as L
import torchvision
from PIL import Image
from einops import rearrange, reduce
from kornia import filter2D
from torch import nn, einsum
from torch.utils.data import Dataset
from torchvision import transforms

from models.stylegan.stylegan2_lucidrains import gradient_penalty
from trainer.networks import register_model
from utils.util import opt_get


def DiffAugment(x, types=[]):
    for p in types:
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x.contiguous()


# """
# Augmentation functions got images as `x`
# where `x` is tensor with this dimensions:
# 0 - count of images
# 1 - channels
# 2 - width
# 3 - height of image
# """

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

def rand_offset(x, ratio=1, ratio_h=1, ratio_v=1):
    w, h = x.size(2), x.size(3)

    imgs = []
    for img in x.unbind(dim = 0):
        max_h = int(w * ratio * ratio_h)
        max_v = int(h * ratio * ratio_v)

        value_h = random.randint(0, max_h) * 2 - max_h
        value_v = random.randint(0, max_v) * 2 - max_v

        if abs(value_h) > 0:
            img = torch.roll(img, value_h, 2)

        if abs(value_v) > 0:
            img = torch.roll(img, value_v, 1)

        imgs.append(img)

    return torch.stack(imgs)

def rand_offset_h(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=ratio, ratio_v=0)

def rand_offset_v(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=0, ratio_v=ratio)

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
    'offset': [rand_offset],
    'offset_h': [rand_offset_h],
    'offset_v': [rand_offset_v],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}

# constants

NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']


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


def is_power_of_two(val):
    return log2(val).is_integer()


def default(val, d):
    return val if exists(val) else d


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


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


def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


def safe_div(n, d):
    try:
        res = n / d
    except ZeroDivisionError:
        prefix = '' if int(n >= 0) else '-'
        res = float(f'{prefix}inf')
    return res


# helper classes

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


class EMAWrapper(nn.Module):
    def __init__(self, wrapped_module, following_module, rate=.995, steps_per_ema=10, steps_per_reset=1000, steps_after_no_reset=25000, reset=True):
        super().__init__()
        self.wrapped = wrapped_module
        self.following = following_module
        self.ema_updater = EMA(rate)
        self.steps_per_ema = steps_per_ema
        self.steps_per_reset = steps_per_reset
        self.steps_after_no_reset = steps_after_no_reset
        if reset:
            self.wrapped.load_state_dict(self.following.state_dict())
        for p in self.wrapped.parameters():
            p.DO_NOT_TRAIN = True

    def reset_parameter_averaging(self):
        self.wrapped.load_state_dict(self.following.state_dict())

    def update_moving_average(self):
        for current_params, ma_params in zip(self.following.parameters(), self.wrapped.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        for current_buffer, ma_buffer in zip(self.following.buffers(), self.wrapped.buffers()):
            new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
            ma_buffer.copy_(new_buffer_value)

    def custom_optimizer_step(self, step):
        if step % self.steps_per_ema == 0:
            self.update_moving_average()
        if step % self.steps_per_reset and step < self.steps_after_no_reset:
            self.reset_parameter_averaging()

    def forward(self, x):
        with torch.no_grad():
            return self.wrapped(x)


class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x):
        return self.g * self.fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SumBranches(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.branches))


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2D(x, f, normalized=True)


# dataset

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class identity(object):
    def __call__(self, tensor):
        return tensor


class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))


def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image


class ImageDataset(Dataset):
    def __init__(
            self,
            folder,
            image_size,
            transparent=False,
            greyscale=False,
            aug_prob=0.
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        if transparent:
            num_channels = 4
            pillow_mode = 'RGBA'
            expand_fn = expand_greyscale(transparent)
        elif greyscale:
            num_channels = 1
            pillow_mode = 'L'
            expand_fn = identity()
        else:
            num_channels = 3
            pillow_mode = 'RGB'
            expand_fn = expand_greyscale(transparent)

        convert_image_fn = partial(convert_image_to, pillow_mode)

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)),
                        transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_fn)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


# augmentations

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))


class AugWrapper(nn.Module):
    def __init__(self, D, image_size, prob, types):
        super().__init__()
        self.D = D
        self.prob = prob
        self.types = types

    def forward(self, images, detach=False, **kwargs):
        context = torch.no_grad if detach else null_context

        with context():
            if random() < self.prob:
                images = random_hflip(images, prob=0.5)
                images = DiffAugment(images, types=self.types)

        return self.D(images, **kwargs)


# modifiable global variables

norm_class = nn.BatchNorm2d


def upsample(scale_factor=2):
    return nn.Upsample(scale_factor=scale_factor)


# squeeze excitation classes

# global context network
# https://arxiv.org/abs/2012.13375
# similar to squeeze-excite, but with a simplified attention pooling and a subsequent layer norm

class GlobalContext(nn.Module):
    def __init__(
            self,
            *,
            chan_in,
            chan_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(chan_in, 1, 1)
        chan_intermediate = max(3, chan_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        context = context.flatten(2).softmax(dim=-1)
        out = einsum('b i n, b c n -> b c i', context, x.flatten(2))
        out = out.unsqueeze(-1)
        return self.net(out)


# frequency channel attention
# https://arxiv.org/abs/2012.11879

def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    return result * (1 if freq == 0 else math.sqrt(2))


def get_dct_weights(width, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, width)
    c_part = channel // len(fidx_u)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for x in range(width):
            for y in range(width):
                coor_value = get_1d_dct(x, u_x, width) * get_1d_dct(y, v_y, width)
                dct_weights[:, i * c_part: (i + 1) * c_part, x, y] = coor_value

    return dct_weights


class FCANet(nn.Module):
    def __init__(
            self,
            *,
            chan_in,
            chan_out,
            reduction=4,
            width
    ):
        super().__init__()

        freq_w, freq_h = ([0] * 8), list(range(8))  # in paper, it seems 16 frequencies was ideal
        dct_weights = get_dct_weights(width, chan_in, [*freq_w, *freq_h], [*freq_h, *freq_w])
        self.register_buffer('dct_weights', dct_weights)

        chan_intermediate = max(3, chan_out // reduction)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = reduce(x * self.dct_weights, 'b c (h h1) (w w1) -> b c h1 w1', 'sum', h1=1, w1=1)
        return self.net(x)


# generative adversarial network

class Generator(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            latent_dim=256,
            fmap_max=512,
            fmap_inverse_coef=12,
            transparent=False,
            greyscale=False,
            freq_chan_attn=False
    ):
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'

        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        fmap_max = default(fmap_max, latent_dim)

        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
            norm_class(latent_dim * 2),
            nn.GLU(dim=1)
        )

        num_layers = int(resolution) - 2
        features = list(map(lambda n: (n, 2 ** (fmap_inverse_coef - n)), range(2, num_layers + 2)))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        features = list(map(lambda n: 3 if n[0] >= 8 else n[1], features))
        features = [latent_dim, *features]

        in_out_features = list(zip(features[:-1], features[1:]))

        self.res_layers = range(2, num_layers + 2)
        self.layers = nn.ModuleList([])
        self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))

        self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
        self.sle_map = list(filter(lambda t: t[0] <= resolution and t[1] <= resolution, self.sle_map))
        self.sle_map = dict(self.sle_map)

        self.num_layers_spatial_res = 1

        for (res, (chan_in, chan_out)) in zip(self.res_layers, in_out_features):
            attn = None
            sle = None
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]

                if freq_chan_attn:
                    sle = FCANet(
                        chan_in=chan_out,
                        chan_out=sle_chan_out,
                        width=2 ** (res + 1)
                    )
                else:
                    sle = GlobalContext(
                        chan_in=chan_out,
                        chan_out=sle_chan_out
                    )

            layer = nn.ModuleList([
                nn.Sequential(
                    upsample(),
                    Blur(),
                    nn.Conv2d(chan_in, chan_out * 2, 3, padding=1),
                    norm_class(chan_out * 2),
                    nn.GLU(dim=1)
                ),
                sle,
                attn
            ])
            self.layers.append(layer)

        self.out_conv = nn.Conv2d(features[-1], init_channel, 3, padding=1)

        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = rearrange(x, 'b c -> b c () ()')
        x = self.initial_conv(x)
        x = F.normalize(x, dim=1)

        residuals = dict()

        for (res, (up, sle, attn)) in zip(self.res_layers, self.layers):
            if exists(attn):
                x = attn(x) + x

            x = up(x)

            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res]

        return self.out_conv(x)


class SimpleDecoder(nn.Module):
    def __init__(
            self,
            *,
            chan_in,
            chan_out=3,
            num_upsamples=4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        final_chan = chan_out
        chans = chan_in

        for ind in range(num_upsamples):
            last_layer = ind == (num_upsamples - 1)
            chan_out = chans if not last_layer else final_chan * 2
            layer = nn.Sequential(
                upsample(),
                nn.Conv2d(chans, chan_out, 3, padding=1),
                nn.GLU(dim=1)
            )
            self.layers.append(layer)
            chans //= 2

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            fmap_max=512,
            fmap_inverse_coef=12,
            transparent=False,
            greyscale=False,
            disc_output_size=5,
            attn_res_layers=[]
    ):
        super().__init__()
        self.image_size = image_size
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'
        assert disc_output_size in {1, 5}, 'discriminator output dimensions can only be 5x5 or 1x1'

        resolution = int(resolution)

        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        num_non_residual_layers = max(0, int(resolution) - 8)
        num_residual_layers = 8 - 3

        non_residual_resolutions = range(min(8, resolution), 2, -1)
        features = list(map(lambda n: (n, 2 ** (fmap_inverse_coef - n)), non_residual_resolutions))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))

        if num_non_residual_layers == 0:
            res, _ = features[0]
            features[0] = (res, init_channel)

        chan_in_out = list(zip(features[:-1], features[1:]))

        self.non_residual_layers = nn.ModuleList([])
        for ind in range(num_non_residual_layers):
            first_layer = ind == 0
            last_layer = ind == (num_non_residual_layers - 1)
            chan_out = features[0][-1] if last_layer else init_channel

            self.non_residual_layers.append(nn.Sequential(
                Blur(),
                nn.Conv2d(init_channel, chan_out, 4, stride=2, padding=1),
                nn.LeakyReLU(0.1)
            ))

        self.residual_layers = nn.ModuleList([])

        for (res, ((_, chan_in), (_, chan_out))) in zip(non_residual_resolutions, chan_in_out):
            attn = None
            self.residual_layers.append(nn.ModuleList([
                SumBranches([
                    nn.Sequential(
                        Blur(),
                        nn.Conv2d(chan_in, chan_out, 4, stride=2, padding=1),
                        nn.LeakyReLU(0.1),
                        nn.Conv2d(chan_out, chan_out, 3, padding=1),
                        nn.LeakyReLU(0.1)
                    ),
                    nn.Sequential(
                        Blur(),
                        nn.AvgPool2d(2),
                        nn.Conv2d(chan_in, chan_out, 1),
                        nn.LeakyReLU(0.1),
                    )
                ]),
                attn
            ]))

        last_chan = features[-1][-1]
        if disc_output_size == 5:
            self.to_logits = nn.Sequential(
                nn.Conv2d(last_chan, last_chan, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(last_chan, 1, 4)
            )
        elif disc_output_size == 1:
            self.to_logits = nn.Sequential(
                Blur(),
                nn.Conv2d(last_chan, last_chan, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(last_chan, 1, 4)
            )

        self.to_shape_disc_out = nn.Sequential(
            nn.Conv2d(init_channel, 64, 3, padding=1),
            Residual(Rezero(GSA(dim=64, norm_queries=True, batch_norm=False))),
            SumBranches([
                nn.Sequential(
                    Blur(),
                    nn.Conv2d(64, 32, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    Blur(),
                    nn.AvgPool2d(2),
                    nn.Conv2d(64, 32, 1),
                    nn.LeakyReLU(0.1),
                )
            ]),
            Residual(Rezero(GSA(dim=32, norm_queries=True, batch_norm=False))),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(32, 1, 4)
        )

        self.decoder1 = SimpleDecoder(chan_in=last_chan, chan_out=init_channel)
        self.decoder2 = SimpleDecoder(chan_in=features[-2][-1], chan_out=init_channel) if resolution >= 9 else None

        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, calc_aux_loss=False):
        orig_img = x

        for layer in self.non_residual_layers:
            x = layer(x)

        layer_outputs = []

        for (net, attn) in self.residual_layers:
            if exists(attn):
                x = attn(x) + x

            x = net(x)
            layer_outputs.append(x)

        out = self.to_logits(x).flatten(1)

        img_32x32 = F.interpolate(orig_img, size=(32, 32))
        out_32x32 = self.to_shape_disc_out(img_32x32)

        if not calc_aux_loss:
            return out, out_32x32, None

        # self-supervised auto-encoding loss

        layer_8x8 = layer_outputs[-1]
        layer_16x16 = layer_outputs[-2]

        recon_img_8x8 = self.decoder1(layer_8x8)

        aux_loss = F.mse_loss(
            recon_img_8x8,
            F.interpolate(orig_img, size=recon_img_8x8.shape[2:])
        )

        if exists(self.decoder2):
            select_random_quadrant = lambda rand_quadrant, img: \
            rearrange(img, 'b c (m h) (n w) -> (m n) b c h w', m=2, n=2)[rand_quadrant]
            crop_image_fn = partial(select_random_quadrant, floor(random() * 4))
            img_part, layer_16x16_part = map(crop_image_fn, (orig_img, layer_16x16))

            recon_img_16x16 = self.decoder2(layer_16x16_part)

            aux_loss_16x16 = F.mse_loss(
                recon_img_16x16,
                F.interpolate(img_part, size=recon_img_16x16.shape[2:])
            )

            aux_loss = aux_loss + aux_loss_16x16

        return out, out_32x32, aux_loss


class LightweightGanDivergenceLoss(L.ConfigurableLoss):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.real = opt['real']
        self.fake = opt['fake']
        self.discriminator = opt['discriminator']
        self.for_gen = opt['gen_loss']
        self.gp_frequency = opt['gradient_penalty_frequency']
        self.noise = opt['noise'] if 'noise' in opt.keys() else 0
        # TODO: Implement generator top-k fractional loss compensation.

    def forward(self, net, state):
        real_input = state[self.real]
        fake_input = state[self.fake]
        if self.noise != 0:
            fake_input = fake_input + torch.rand_like(fake_input) * self.noise
            real_input = real_input + torch.rand_like(real_input) * self.noise

        D = self.env['discriminators'][self.discriminator]
        fake, fake32, _ = D(fake_input, detach=not self.for_gen)
        if self.for_gen:
            return fake.mean() + fake32.mean()
        else:
            real_input.requires_grad_()  # <-- Needed to compute gradients on the input.
            real, real32, real_aux = D(real_input, calc_aux_loss=True)
            divergence_loss = hinge_loss(real, fake) + hinge_loss(real32, fake32) + real_aux

            # Apply gradient penalty. TODO: migrate this elsewhere.
            if self.env['step'] % self.gp_frequency == 0:
                gp = gradient_penalty(real_input, real)
                self.metrics.append(("gradient_penalty", gp.clone().detach()))
                divergence_loss = divergence_loss + gp

            real_input.requires_grad_(requires_grad=False)
            return divergence_loss


@register_model
def register_lightweight_gan_g(opt_net, opt, other_nets):
    gen = Generator(**opt_net['kwargs'])
    if opt_get(opt_net, ['ema'], False):
        following = other_nets[opt_net['following']]
        return EMAWrapper(gen, following, opt_net['rate'])
    return gen


@register_model
def register_lightweight_gan_d(opt_net, opt):
    d = Discriminator(**opt_net['kwargs'])
    if opt_net['aug']:
        return AugWrapper(d, d.image_size, opt_net['aug_prob'], opt_net['aug_types'])
    return d


if __name__ == '__main__':
    g = Generator(image_size=256)
    d = Discriminator(image_size=256)
    j = torch.randn(1,256)
    r = g(j)
    a, b, c = d(r)
    print(a.shape)
