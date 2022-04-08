import random
from time import time

import kornia
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
from kornia import augmentation as augs, geometry
from kornia import filters
import torch.nn as nn
import torch.nn.functional as F

# Wrapper for a DLAS Dataset class that applies random augmentations from the BYOL paper to BOTH the 'lq' and 'hq'
# inputs. These are then outputted as 'aug1' and 'aug2'.
from tqdm import tqdm

from data import create_dataset
from models.arch_util import PixelUnshuffle
from utils.util import opt_get


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
        self.key1 = opt_get(opt, ['key1'], 'hq')
        self.key2 = opt_get(opt, ['key2'], 'lq')
        for_sr = opt_get(opt, ['for_sr'], False)  # When set, color alterations and blurs are disabled.

        augmentations = [ \
            augs.RandomHorizontalFlip(),
            augs.RandomResizedCrop((self.cropped_img_size, self.cropped_img_size))]
        if not for_sr:
            augmentations.extend([RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
                                  augs.RandomGrayscale(p=0.2),
                                  RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1)])
        if opt['normalize']:
            # The paper calls for normalization. Most datasets/models in this repo don't use this.
            # Recommend setting true if you want to train exactly like the paper.
            augmentations.append(augs.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])))
        self.aug = nn.Sequential(*augmentations)

    def __getitem__(self, item):
        item = self.wrapped_dataset[item]
        item.update({'aug1': self.aug(item[self.key1]).squeeze(dim=0), 'aug2': self.aug(item[self.key2]).squeeze(dim=0)})
        return item

    def __len__(self):
        return len(self.wrapped_dataset)


# Basically the same as ByolDatasetWrapper except only produces 1 augmentation and stores in the 'lr' key. Also applies
# crop&resize to 2D tensors in the state dict with the word "label" in them.
class DatasetRandomAugWrapper(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.wrapped_dataset = create_dataset(opt['dataset'])
        self.cropped_img_size = opt['crop_size']
        self.includes_labels = opt['includes_labels']
        augmentations = [ \
            RandomApply(augs.ColorJitter(0.4, 0.4, 0.4, 0.2), p=0.8),
            augs.RandomGrayscale(p=0.2),
            RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1)]
        self.aug = nn.Sequential(*augmentations)
        self.rrc = nn.Sequential(*[
            augs.RandomHorizontalFlip(),
            augs.RandomResizedCrop((self.cropped_img_size, self.cropped_img_size))])

    def __getitem__(self, item):
        item = self.wrapped_dataset[item]
        hq = self.aug(item['hq'].unsqueeze(0)).squeeze(0)
        labels = []
        dtypes = []
        for k in item.keys():
            if 'label' in k and isinstance(item[k], torch.Tensor) and len(item[k].shape) == 3:
                assert item[k].shape[0] == 1   # Only supports a channel dim of 1.
                labels.append(k)
                dtypes.append(item[k].dtype)
                hq = torch.cat([hq, item[k].type(torch.float)], dim=0)
        hq = self.rrc(hq.unsqueeze(0)).squeeze(0)
        for i, k in enumerate(labels):
            # Strip out any label values that are not whole numbers.
            item[k] = hq[3+i:3+i+1,:,:]
            whole = (item[k].round() == item[k])
            item[k] = item[k] * whole
            item[k] = item[k].type(dtypes[i])
        item['lq'] = hq[:3,:,:]
        item['hq'] = item['lq']
        return item

    def __len__(self):
        return len(self.wrapped_dataset)


def test_dataset_random_aug_wrapper():
    opt = {
        'dataset': {
            'mode': 'imagefolder',
            'name': 'amalgam',
            'paths': ['F:\\4k6k\\datasets\\ns_images\\512_unsupervised'],
            'weights': [1],
            'target_size': 512,
            'force_multiple': 1,
            'scale': 1,
            'fixed_corruptions': ['jpeg-broad'],
            'random_corruptions': ['noise-5', 'none'],
            'num_corrupts_per_image': 1,
            'corrupt_before_downsize': False,
            'labeler': {
                'type': 'patch_labels',
                'label_file': 'F:\\4k6k\\datasets\\ns_images\\512_unsupervised\\categories.json'
            }
        },
        'crop_size': 512,
        'includes_labels': True,
    }

    ds = DatasetRandomAugWrapper(opt)
    import os
    os.makedirs("debug", exist_ok=True)
    for i in tqdm(range(0, len(ds))):
        o = ds[random.randint(0, len(ds)-1)]
        for k, v in o.items():
            # 'lq', 'hq', 'aug1', 'aug2',
            if k in ['hq']:
                torchvision.utils.save_image(v.unsqueeze(0), "debug/%i_%s.png" % (i, k))
                masked = v * (o['labels_mask'] * .5 + .5)
                #torchvision.utils.save_image(masked.unsqueeze(0), "debug/%i_%s_masked.png" % (i, k))
                # Pick a random (non-zero) label and spit it out with the textual label.
                if len(o['labels'].unique()) > 1:
                    randlbl = np.random.choice(o['labels'].unique()[1:])
                    moremask = v * ((1*(o['labels'] == randlbl))*.5+.5)
                    torchvision.utils.save_image(moremask.unsqueeze(0), "debug/%i_%s_%s.png" % (i, k, o['label_strings'][randlbl]))


def no_batch_interpolate(i, size, mode):
    i = i.unsqueeze(0)
    i = F.interpolate(i, size=size, mode=mode)
    return i.squeeze(0)


# Performs a 1-d translation of "other":
# If other<ref, returns 0.
# Else: return other-ref
def snap(ref, other):
    if other < ref:
        return 0
    return other - ref


# Pads a tensor with zeros so that it fits in a dxd square.
def pad_to(im, d):
    if len(im.shape) == 3:
        pd = torch.zeros((im.shape[0],d,d))
        pd[:, :im.shape[1], :im.shape[2]] = im
    else:
        pd = torch.zeros((im.shape[0],im.shape[1],d,d), device=im.device)
        pd[:, :, :im.shape[2], :im.shape[3]] = im
    return pd


# Variation of RandomResizedCrop, which picks a region of the image that the two augments must share. The augments
# then propagate off random corners of the shared region, using the same scale.
#
# Operates in units of "multiple". The intent is that this multiple is equivalent to the compression multiple of the
# latent space being used so that each structural unit corresponds to a latent unit.
class RandomSharedRegionCrop(nn.Module):
    def __init__(self, multiple, jitter_range=0):
        super().__init__()
        self.multiple = multiple
        self.jitter_range = jitter_range  # When specified, images are shifted an additional random([-j,j]) pixels where j=jitter_range

    def forward(self, i1, i2):
        assert i1.shape[-1] == i2.shape[-1]
        # Outline of the general algorithm:
        # 1. Assume the input is a square. Divide it by self.multiple to get working units.
        # 2. Pick a random width, height and top corner location for the first patch.
        # 3. Pick a random width, height and top corner location for the second patch.
        #    Note: All dims from (2) and (3) must contain at least half of the image, guaranteeing overlap.
        # 4. Build patches from input images. Resize them appropriately. Apply translational jitter.\
        # 5. Randomly flip image 2 if needed.
        # 5. Compute the metrics needed to extract overlapping regions from the resized patches: top, left,
        #    original_height, original_width.
        # 6. Compute the "shared_view" from the above data.

        # Step 1
        c, d, _ = i1.shape
        assert d % self.multiple == 0 and d > (self.multiple*3)
        d = d // self.multiple

        # Step 2
        base_w = random.randint(d//2+1, d-1)
        base_l = random.randint(0, d-base_w)
        base_h = random.randint(base_w-1, base_w+1)
        base_t = random.randint(0, d-base_h)
        base_r, base_b = base_l+base_w, base_t+base_h

        # Step 3
        im2_w = random.randint(d//2+1, d-1)
        im2_l = random.randint(0, d-im2_w)
        im2_h = random.randint(im2_w-1, im2_w+1)
        im2_t = random.randint(0, d-im2_h)
        im2_r, im2_b = im2_l+im2_w, im2_t+im2_h

        # Step 4
        m = self.multiple
        jl, jt = random.randint(-self.jitter_range, self.jitter_range), random.randint(-self.jitter_range, self.jitter_range)
        jt = jt if base_t != 0 else abs(jt)  # If the top of a patch is zero, a negative jitter will cause it to go negative.
        jt = jt if (base_t+base_h)*m != i1.shape[1] else 0 # Likewise, jitter shouldn't allow the patch to go over-bounds.
        jl = jl if base_l != 0 else abs(jl)
        jl = jl if (base_l+base_w)*m != i1.shape[1] else 0
        p1 = i1[:, base_t*m+jt:(base_t+base_h)*m+jt, base_l*m+jl:(base_l+base_w)*m+jl]
        p1_resized = no_batch_interpolate(p1, size=(d*m, d*m), mode="bilinear")
        jl, jt = random.randint(-self.jitter_range, self.jitter_range), random.randint(-self.jitter_range, self.jitter_range)
        jt = jt if im2_t != 0 else abs(jt)
        jt = jt if (im2_t+im2_h)*m != i2.shape[1] else 0
        jl = jl if im2_l != 0 else abs(jl)
        jl = jl if (im2_l+im2_w)*m != i2.shape[1] else 0
        p2 = i2[:, im2_t*m+jt:(im2_t+im2_h)*m+jt, im2_l*m+jl:(im2_l+im2_w)*m+jl]
        p2_resized = no_batch_interpolate(p2, size=(d*m, d*m), mode="bilinear")

        # Step 5
        should_flip = random.random() < .5
        if should_flip:
            should_flip = 1
            p2_resized = geometry.transform.hflip(p2_resized)
        else:
            should_flip = 0

        # Step 6
        i1_shared_t, i1_shared_l = snap(base_t, im2_t), snap(base_l, im2_l)
        i2_shared_t, i2_shared_l = snap(im2_t, base_t), snap(im2_l, base_l)
        ix_h = min(base_b, im2_b) - max(base_t, im2_t)
        ix_w = min(base_r, im2_r) - max(base_l, im2_l)
        recompute_package = torch.tensor([d, base_h, base_w, i1_shared_t, i1_shared_l, im2_h, im2_w, i2_shared_t, i2_shared_l, should_flip, ix_h, ix_w], dtype=torch.long)

        # Step 7
        mask1 = torch.full((1, base_h*m, base_w*m), fill_value=.5)
        mask1[:, i1_shared_t*m:(i1_shared_t+ix_h)*m, i1_shared_l*m:(i1_shared_l+ix_w)*m] = 1
        masked1 = pad_to(p1 * mask1, d*m)
        mask2 = torch.full((1, im2_h*m, im2_w*m), fill_value=.5)
        mask2[:, i2_shared_t*m:(i2_shared_t+ix_h)*m, i2_shared_l*m:(i2_shared_l+ix_w)*m] = 1
        masked2 = pad_to(p2 * mask2, d*m)
        mask = torch.full((1, d*m, d*m), fill_value=.33)
        mask[:, base_t*m:(base_t+base_w)*m, base_l*m:(base_l+base_h)*m] += .33
        mask[:, im2_t*m:(im2_t+im2_w)*m, im2_l*m:(im2_l+im2_h)*m] += .33
        masked_dbg = i1 * mask

        # Step 8 - Rebuild shared regions for testing purposes.
        p1_shuf, p2_shuf = PixelUnshuffle(self.multiple)(p1_resized.unsqueeze(0)), \
                           PixelUnshuffle(self.multiple)(p2_resized.unsqueeze(0))
        i1_shared, i2_shared = reconstructed_shared_regions(p1_shuf, p2_shuf, recompute_package.unsqueeze(0))
        i1_shared = pad_to(nn.PixelShuffle(self.multiple)(i1_shared).squeeze(0), d * m)
        i2_shared = pad_to(nn.PixelShuffle(self.multiple)(i2_shared).squeeze(0), d*m)

        return p1_resized, p2_resized, recompute_package, masked1, masked2, masked_dbg, i1_shared, i2_shared


# Uses the recompute package returned from the above dataset to extract matched-size "similar regions" from two feature
# maps.
def reconstructed_shared_regions(fea1, fea2, recompute_package: torch.Tensor):
    package = recompute_package.cpu()
    res1 = []
    res2 = []
    pad_dim = torch.max(package[:, -2:]).item()
    # It'd be real nice if we could do this at the batch level, but I don't see a really good way to do that outside
    # of conforming the recompute_package across the entire batch.
    for b in range(package.shape[0]):
        expected_dim, f1_h, f1_w, f1s_t, f1s_l, f2_h, f2_w, f2s_t, f2s_l, should_flip, s_h, s_w = tuple(package[b].tolist())
        # If you are hitting this assert, you specified `latent_multiple` in your dataset config wrong.
        assert expected_dim == fea1.shape[2] and expected_dim == fea2.shape[2]

        # Unflip 2 if needed.
        f2 = fea2[b]
        if should_flip == 1:
            f2 = kornia.geometry.transform.hflip(f2)
        # Resize the input features to match
        f1s = F.interpolate(fea1[b].unsqueeze(0), (f1_h, f1_w), mode="nearest")
        f2s = F.interpolate(f2.unsqueeze(0), (f2_h, f2_w), mode="nearest")
        # Outputs must be padded so they can "get along" with each other.
        res1.append(pad_to(f1s[:, :, f1s_t:f1s_t+s_h, f1s_l:f1s_l+s_w], pad_dim))
        res2.append(pad_to(f2s[:, :, f2s_t:f2s_t+s_h, f2s_l:f2s_l+s_w], pad_dim))
    return torch.cat(res1, dim=0), torch.cat(res2, dim=0)


# Follows the general template of BYOL dataset, with the following changes:
# 1. Flip() is not applied.
# 2. Instead of RandomResizedCrop, a custom Transform, RandomSharedRegionCrop is used.
# 3. The dataset injects two integer tensors alongside the augmentations, which are used to index image regions shared
#    by the joint augmentations.
# 4. The dataset injects an aug_shared_view for debugging purposes.
class StructuredCropDatasetWrapper(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.wrapped_dataset = create_dataset(opt['dataset'])
        augmentations = [RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
            augs.RandomGrayscale(p=0.2),
            RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1)]
        self.aug = nn.Sequential(*augmentations)
        self.rrc = RandomSharedRegionCrop(opt['latent_multiple'], opt_get(opt, ['jitter_range'], 0))

    def __getitem__(self, item):
        item = self.wrapped_dataset[item]
        a1 = self.aug(item['hq']).squeeze(dim=0)
        a2 = self.aug(item['lq']).squeeze(dim=0)
        a1, a2, sr_dim, m1, m2, db, i1s, i2s = self.rrc(a1, a2)
        item.update({'aug1': a1, 'aug2': a2, 'similar_region_dimensions': sr_dim,
                     'masked1': m1, 'masked2': m2, 'aug_shared_view': db,
                     'i1_shared': i1s, 'i2_shared': i2s})
        return item

    def __len__(self):
        return len(self.wrapped_dataset)


# For testing this dataset.
def test_structured_crop_dataset_wrapper():
    opt = {
        'dataset':
            {
            'mode': 'imagefolder',
            'name': 'amalgam',
            'paths': ['F:\\4k6k\\datasets\\ns_images\\512_unsupervised'],
            'weights': [1],
            'target_size': 256,
            'force_multiple': 32,
            'scale': 1,
            'fixed_corruptions': ['jpeg-broad', 'gaussian_blur'],
            'random_corruptions': ['noise-5', 'none'],
            'num_corrupts_per_image': 1,
            'corrupt_before_downsize': True,
            },
        'latent_multiple': 16,
        'jitter_range': 0,
    }

    ds = StructuredCropDatasetWrapper(opt)
    import os
    os.makedirs("debug", exist_ok=True)
    for i in tqdm(range(0, len(ds))):
        o = ds[random.randint(0, len(ds)-1)]
        #for k, v in o.items():
            # 'lq', 'hq', 'aug1', 'aug2',
            #if k in [ 'aug_shared_view', 'masked1', 'masked2']:
                #torchvision.utils.save_image(v.unsqueeze(0), "debug/%i_%s.png" % (i, k))
        rcpkg = o['similar_region_dimensions']
        pixun = PixelUnshuffle(16)
        pixsh = nn.PixelShuffle(16)
        rc1, rc2 = reconstructed_shared_regions(pixun(o['aug1'].unsqueeze(0)), pixun(o['aug2'].unsqueeze(0)), rcpkg.unsqueeze(0))
        #torchvision.utils.save_image(pixsh(rc1), "debug/%i_rc1.png" % (i,))
        #torchvision.utils.save_image(pixsh(rc2), "debug/%i_rc2.png" % (i,))


if __name__ == '__main__':
    test_dataset_random_aug_wrapper()
