import random
import cv2
import numpy as np
from data.util import read_img
from PIL import Image
from io import BytesIO

# Performs image corruption on a list of images from a configurable set of corruption
# options.
class ImageCorruptor:
    def __init__(self, opt):
        self.fixed_corruptions = opt['fixed_corruptions']
        self.num_corrupts = opt['num_corrupts_per_image'] if 'num_corrupts_per_image' in opt.keys() else 2
        if self.num_corrupts == 0:
            return
        self.random_corruptions = opt['random_corruptions']
        self.blur_scale = opt['corruption_blur_scale'] if 'corruption_blur_scale' in opt.keys() else 1

    def corrupt_images(self, imgs):
        if self.num_corrupts == 0 and not self.fixed_corruptions:
            return imgs

        if self.num_corrupts == 0:
            augmentations = []
        else:
            augmentations = random.choices(self.random_corruptions, k=self.num_corrupts)
        # Source of entropy, which should be used across all images.
        rand_int_f = random.randint(1, 999999)
        rand_int_a = random.randint(1, 999999)

        corrupted_imgs = []
        applied_augs = augmentations + self.fixed_corruptions
        for img in imgs:
            for aug in augmentations:
                img = self.apply_corruption(img, aug, rand_int_a, applied_augs)
            for aug in self.fixed_corruptions:
                img = self.apply_corruption(img, aug, rand_int_f, applied_augs)
            corrupted_imgs.append(img)

        return corrupted_imgs

    def apply_corruption(self, img, aug, rand_int, applied_augmentations):
        if 'color_quantization' in aug:
            # Color quantization
            quant_div = 2 ** ((rand_int % 3) + 2)
            img = img * 255
            img = (img // quant_div) * quant_div
            img = img / 255
        elif 'gaussian_blur' in aug:
            # Gaussian Blur
            if aug == 'gaussian_blur_3':
                kernel = 3
            else:
                kernel = 2 * self.blur_scale * (rand_int % 3) + 1
            img = cv2.GaussianBlur(img, (kernel, kernel), 3)
        elif 'motion_blur' in aug:
            # Motion blur
            intensity = self.blur_scale * (rand_int % 3) + 1
            angle = (rand_int // 3) % 360
            k = np.zeros((intensity, intensity), dtype=np.float32)
            k[(intensity - 1) // 2, :] = np.ones(intensity, dtype=np.float32)
            k = cv2.warpAffine(k, cv2.getRotationMatrix2D((intensity / 2 - 0.5, intensity / 2 - 0.5), angle, 1.0),
                               (intensity, intensity))
            k = k * (1.0 / np.sum(k))
            img = cv2.filter2D(img, -1, k)
        elif 'smooth_blur' in aug:
            # Smooth blur
            kernel = 2 * self.blur_scale * (rand_int % 3) + 1
            img = cv2.blur(img, ksize=(kernel, kernel))
        elif 'block_noise' in aug:
            # Large distortion blocks in part of an img, such as is used to mask out a face.
            pass
        elif 'lq_resampling' in aug:
            # Bicubic LR->HR
            pass
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
                noise_intensity = (rand_int % 4 + 2) / 255.0  # Between 1-4
            img += np.random.randn(*img.shape) * noise_intensity
        elif 'jpeg' in aug:
            if 'noise' not in applied_augmentations and 'noise-5' not in applied_augmentations:
                if aug == 'jpeg':
                    lo=10
                    range=20
                elif aug == 'jpeg-medium':
                    lo=23
                    range=25
                elif aug == 'jpeg-broad':
                    lo=15
                    range=60
                # JPEG compression
                qf = (rand_int % range + lo)
                # cv2's jpeg compression is "odd". It introduces artifacts. Use PIL instead.
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                buffer = BytesIO()
                img.save(buffer, "JPEG", quality=qf, optimize=True)
                buffer.seek(0)
                jpeg_img_bytes = np.asarray(bytearray(buffer.read()), dtype="uint8")
                img = read_img("buffer", jpeg_img_bytes, rgb=True)
        elif 'saturation' in aug:
            # Lightening / saturation
            saturation = float(rand_int % 10) * .03
            img = np.clip(img + saturation, a_max=1, a_min=0)
        elif 'none' not in aug:
            raise NotImplementedError("Augmentation doesn't exist")

        return img
