import glob
import itertools
import random

import cv2
import numpy as np
import torch
import os

from data import util
# Builds a dataset created from a simple folder containing a list of training/test/validation images.
from data.image_corruptor import ImageCorruptor


class ImageFolderDataset:
    def __init__(self, opt):
        self.opt = opt
        self.corruptor = ImageCorruptor(opt)
        self.target_hq_size = opt['target_size'] if 'target_size' in opt.keys() else None
        self.multiple = opt['force_multiple'] if 'force_multiple' in opt.keys() else 1
        self.scale = opt['scale']
        self.paths = opt['paths']
        self.corrupt_before_downsize = opt['corrupt_before_downsize'] if 'corrupt_before_downsize' in opt.keys() else False
        assert (self.target_hq_size // self.scale) % self.multiple == 0  # If we dont throw here, we get some really obscure errors.
        if not isinstance(self.paths, list):
            self.paths = [self.paths]
            self.weights = [1]
        else:
            self.weights = opt['weights']

        # Just scan the given directory for images of standard types.
        supported_types = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'gif', 'GIF']
        self.image_paths = []
        for path, weight in zip(self.paths, self.weights):
            cache_path = os.path.join(path, 'cache.pth')
            if os.path.exists(cache_path):
                imgs = torch.load(cache_path)
            else:
                print("Building image folder cache, this can take some time for large datasets..")
                imgs = []
                for ext in supported_types:
                    imgs.extend(glob.glob(os.path.join(path, "*." + ext)))
                torch.save(imgs, cache_path)
            for w in range(weight):
                self.image_paths.extend(imgs)
        self.len = len(self.image_paths)

    def get_paths(self):
        return self.image_paths

    # Given an HQ square of arbitrary size, resizes it to specifications from opt.
    def resize_hq(self, imgs_hq):
        # Enforce size constraints
        h, w, _ = imgs_hq[0].shape
        if self.target_hq_size is not None and self.target_hq_size != h:
            hqs_adjusted = []
            for hq in imgs_hq:
                # It is assumed that the target size is a square.
                target_size = (self.target_hq_size, self.target_hq_size)
                hqs_adjusted.append(cv2.resize(hq, target_size, interpolation=cv2.INTER_AREA))
            h, w = self.target_hq_size, self.target_hq_size
        else:
            hqs_adjusted = imgs_hq
        hq_multiple = self.multiple * self.scale  # Multiple must apply to LQ image.
        if h % hq_multiple != 0 or w % hq_multiple != 0:
            hqs_conformed = []
            for hq in hqs_adjusted:
                h, w = (h - h % hq_multiple), (w - w % hq_multiple)
                hqs_conformed.append(hq[:h, :w, :])
            return hqs_conformed
        return hqs_adjusted

    def synthesize_lq(self, hs):
        h, w, _ = hs[0].shape
        ls = []
        if self.corrupt_before_downsize:
            hs = self.corruptor.corrupt_images(hs)
        for hq in hs:
            ls.append(cv2.resize(hq, (h // self.scale, w // self.scale), interpolation=cv2.INTER_AREA))
        # Corrupt the LQ image (only in eval mode)
        if not self.corrupt_before_downsize:
            ls = self.corruptor.corrupt_images(ls)
        return ls

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        hq = util.read_img(None, self.image_paths[item], rgb=True)

        hs = self.resize_hq([hq])
        ls = self.synthesize_lq(hs)

        # Convert to torch tensor
        hq = torch.from_numpy(np.ascontiguousarray(np.transpose(hs[0], (2, 0, 1)))).float()
        lq = torch.from_numpy(np.ascontiguousarray(np.transpose(ls[0], (2, 0, 1)))).float()

        return {'lq': lq, 'hq': hq, 'LQ_path': self.image_paths[item], 'HQ_path': self.image_paths[item]}


if __name__ == '__main__':
    opt = {
        'name': 'amalgam',
        'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\random_100_1024px'],
        'weights': [1],
        'target_size': 128,
        'force_multiple': 32,
        'scale': 2,
        'fixed_corruptions': ['jpeg-broad', 'gaussian_blur'],
        'random_corruptions': ['noise-5', 'none'],
        'num_corrupts_per_image': 1,
        'corrupt_before_downsize': True,
    }

    ds = ImageFolderDataset(opt)
    import os
    os.makedirs("debug", exist_ok=True)
    for i in range(0, len(ds)):
        o = ds[random.randint(0, len(ds))]
        #for k, v in o.items():
        k = 'lq'
        v = o[k]
        #if 'LQ' in k and 'path' not in k and 'center' not in k:
        #if 'full' in k:
            #masked = v[:3, :, :] * v[3]
            #torchvision.utils.save_image(masked.unsqueeze(0), "debug/%i_%s_masked.png" % (i, k))
            #v = v[:3, :, :]
        import torchvision
        torchvision.utils.save_image(v.unsqueeze(0), "debug/%i_%s.png" % (i, k))