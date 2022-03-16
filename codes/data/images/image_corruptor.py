import functools
import random
from math import cos, pi

import cv2
import kornia
import numpy as np
import torch
from kornia.augmentation import ColorJitter

from data.util import read_img
from PIL import Image
from io import BytesIO


# Get a rough visualization of the above distribution. (Y-axis is meaningless, just spreads data)
from utils.util import opt_get

'''
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    data = np.asarray([get_rand() for _ in range(5000)])
    plt.plot(data, np.random.uniform(size=(5000,)), 'x')
    plt.show()
'''


def kornia_color_jitter_numpy(img, setting):
    if setting * 255 > 1:
        # I'm using Kornia's ColorJitter, which requires pytorch arrays in b,c,h,w format.
        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
        img = ColorJitter(setting, setting, setting, setting)(img)
        img = img.squeeze(0).permute(1,2,0).numpy()
    return img


# Performs image corruption on a list of images from a configurable set of corruption
# options.
class ImageCorruptor:
    def __init__(self, opt):
        self.opt = opt
        self.reset_random()
        self.blur_scale = opt['corruption_blur_scale'] if 'corruption_blur_scale' in opt.keys() else 1
        self.fixed_corruptions = opt['fixed_corruptions'] if 'fixed_corruptions' in opt.keys() else []
        self.num_corrupts = opt['num_corrupts_per_image'] if 'num_corrupts_per_image' in opt.keys() else 0
        self.cosine_bias = opt_get(opt, ['cosine_bias'], True)
        if self.num_corrupts == 0:
            return
        else:
            self.random_corruptions = opt['random_corruptions'] if 'random_corruptions' in opt.keys() else []

    def reset_random(self):
        if 'random_seed' in self.opt.keys():
            self.rand = random.Random(self.opt['random_seed'])
        else:
            self.rand = random.Random()

    # Feeds a random uniform through a cosine distribution to slightly bias corruptions towards "uncorrupted".
    # Return is on [0,1] with a bias towards 0.
    def get_rand(self):
        r = self.rand.random()
        if self.cosine_bias:
            return 1 - cos(r * pi / 2)
        else:
            return r

    def corrupt_images(self, imgs, return_entropy=False):
        if self.num_corrupts == 0 and not self.fixed_corruptions:
            if return_entropy:
                return imgs, []
            else:
                return imgs

        if self.num_corrupts == 0:
            augmentations = []
        else:
            augmentations = random.choices(self.random_corruptions, k=self.num_corrupts)

        # Sources of entropy
        corrupted_imgs = []
        entropy = []
        undo_fns = []
        applied_augs = augmentations + self.fixed_corruptions
        for img in imgs:
            for aug in augmentations:
                r = self.get_rand()
                img, undo_fn = self.apply_corruption(img, aug, r, applied_augs)
                if undo_fn is not None:
                    undo_fns.append(undo_fn)
            for aug in self.fixed_corruptions:
                r = self.get_rand()
                img, undo_fn = self.apply_corruption(img, aug, r, applied_augs)
                entropy.append(r)
                if undo_fn is not None:
                    undo_fns.append(undo_fn)
            # Apply undo_fns after all corruptions are finished, in same order.
            for ufn in undo_fns:
                img = ufn(img)
            corrupted_imgs.append(img)


        if return_entropy:
            return corrupted_imgs, entropy
        else:
            return corrupted_imgs

    def apply_corruption(self, img, aug, rand_val, applied_augmentations):
        undo_fn = None
        if 'color_quantization' in aug:
            # Color quantization
            quant_div = 2 ** (int(rand_val * 10 / 3) + 2)
            img = img * 255
            img = (img // quant_div) * quant_div
            img = img / 255
        elif 'color_jitter' in aug:
            lo_end = 0
            hi_end = .2
            setting = rand_val * (hi_end - lo_end) + lo_end
            img = kornia_color_jitter_numpy(img, setting)
        elif 'gaussian_blur' in aug:
            img = cv2.GaussianBlur(img, (0,0), self.blur_scale*rand_val*1.5)
        elif 'motion_blur' in aug:
            # Motion blur
            intensity = self.blur_scale*rand_val * 3 + 1
            angle = random.randint(0,360)
            k = np.zeros((intensity, intensity), dtype=np.float32)
            k[(intensity - 1) // 2, :] = np.ones(intensity, dtype=np.float32)
            k = cv2.warpAffine(k, cv2.getRotationMatrix2D((intensity / 2 - 0.5, intensity / 2 - 0.5), angle, 1.0),
                               (intensity, intensity))
            k = k * (1.0 / np.sum(k))
            img = cv2.filter2D(img, -1, k)
        elif 'block_noise' in aug:
            # Large distortion blocks in part of an img, such as is used to mask out a face.
            pass
        elif 'lq_resampling' in aug:
            # Random mode interpolation HR->LR->HR
            if 'lq_resampling4x' == aug:
                scale = 4
            else:
                if rand_val < .3:
                    scale = 1
                elif rand_val < .7:
                    scale = 2
                else:
                    scale = 4
            if scale > 1:
                interpolation_modes = [cv2.INTER_NEAREST, cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_LANCZOS4]
                mode = random.randint(0,4) % len(interpolation_modes)
                # Downsample first, then upsample using the random mode.
                img = cv2.resize(img, dsize=(img.shape[1]//scale, img.shape[0]//scale), interpolation=mode)
                def lq_resampling_undo_fn(scale, img):
                    return cv2.resize(img, dsize=(img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_LINEAR)
                undo_fn = functools.partial(lq_resampling_undo_fn, scale)
        elif 'color_shift' in aug:
            # Color shift
            pass
        elif 'interlacing' in aug:
            # Interlacing distortion
            pass
        elif 'chromatic_aberration' in aug:
            # Chromatic aberration
            pass
        elif 'noise' in aug:
            # Random noise
            if 'noise-5' == aug:
                noise_intensity = 5 / 255.0
            else:
                noise_intensity = (rand_val*6) / 255.0
            img += np.random.rand(*img.shape) * noise_intensity
        elif 'jpeg' in aug:
            if 'noise' not in applied_augmentations and 'noise-5' not in applied_augmentations:
                if aug == 'jpeg':
                    lo=10
                    range=20
                elif aug == 'jpeg-low':
                    lo=15
                    range=10
                elif aug == 'jpeg-medium':
                    lo=23
                    range=25
                elif aug == 'jpeg-broad':
                    lo=15
                    range=60
                elif aug == 'jpeg-normal':
                    lo=47
                    range=35
                else:
                    raise NotImplementedError("specified jpeg corruption doesn't exist")
                # JPEG compression
                qf = (int((1-rand_val)*range) + lo)
                # Use PIL to perform a mock compression to a data buffer, then swap back to cv2.
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                buffer = BytesIO()
                img.save(buffer, "JPEG", quality=qf, optimize=True)
                buffer.seek(0)
                jpeg_img_bytes = np.asarray(bytearray(buffer.read()), dtype="uint8")
                img = read_img("buffer", jpeg_img_bytes, rgb=True)
        elif 'saturation' in aug:
            # Lightening / saturation
            saturation = rand_val * .3
            img = np.clip(img + saturation, a_max=1, a_min=0)
        elif 'greyscale' in aug:
            img = np.tile(np.mean(img, axis=2, keepdims=True), [1,1,3])
        elif 'none' not in aug:
            raise NotImplementedError("Augmentation doesn't exist")

        return img, undo_fn
