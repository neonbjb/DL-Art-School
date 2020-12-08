import random

import torch
from torch.utils.data import Dataset
from kornia import augmentation as augs
from kornia import filters
import torch.nn as nn

# Wrapper for a DLAS Dataset class that applies random augmentations from the BYOL paper to BOTH the 'lq' and 'hq'
# inputs. These are then outputted as 'aug1' and 'aug2'.
from data import create_dataset


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class ByolDatasetWrapper(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.wrapped_dataset = create_dataset(opt['dataset'])
        self.cropped_img_size = opt['crop_size']
        augmentations = [ \
            RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
            augs.RandomGrayscale(p=0.2),
            augs.RandomHorizontalFlip(),
            RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
            augs.RandomResizedCrop((self.cropped_img_size, self.cropped_img_size))]
        if opt['normalize']:
            # The paper calls for normalization. Recommend setting true if you want exactly like the paper.
            augmentations.append(augs.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])))
        self.aug = nn.Sequential(*augmentations)

    def __getitem__(self, item):
        item = self.wrapped_dataset[item]
        item.update({'aug1': self.aug(item['hq']).squeeze(dim=0), 'aug2': self.aug(item['lq']).squeeze(dim=0)})
        return item

    def __len__(self):
        return len(self.wrapped_dataset)
