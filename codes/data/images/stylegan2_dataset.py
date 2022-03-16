from functools import partial
from random import random

import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms
import torch.nn as nn
from pathlib import Path

import models.image_generation.stylegan.stylegan2_lucidrains as sg2


def convert_transparent_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def convert_rgb_to_transparent(image):
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image


def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)


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

        if not sg2.exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))


class Stylegan2Dataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        EXTS = ['jpg', 'jpeg', 'png', 'webp']
        self.folder = opt['path']
        self.image_size = opt['target_size']
        self.paths = [p for ext in EXTS for p in Path(f'{self.folder}').glob(f'**/*.{ext}')]
        aug_prob = opt['aug_prob']
        transparent = opt['transparent'] if 'transparent' in opt.keys() else False
        assert len(self.paths) > 0, f'No images were found in {self.folder} for training'

        convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, self.image_size)),
            transforms.Resize(self.image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(self.image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)),
                        transforms.CenterCrop(self.image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(transparent))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = self.transform(img)
        return {'lq': img, 'hq': img, 'GT_path': str(path)}
