import random


# Performs image corruption on a list of images from a configurable set of corruption
# options.
class ImageCorruptor:
    def __init__(self, opt):
        self.num_corrupts = opt['num_corrupts_per_image'] if 'num_corrupts_per_image' in opt.keys() else 2
        self.corruptions_enabled = opt['corruptions']

    def corrupt_images(self, imgs):
        augmentations = random.choice(self.corruptions_enabled, k=self.num_corrupts)
        # Source of entropy, which should be used across all images.
        rand_int = random.randint(1, 999999)

        corrupted_imgs = []
        for img in imgs:
            for aug in augmentations:
                if 'color_quantization' in aug:
                    # Color quantization
                    quant_div = 2 ** random.randint(1, 4)
                    augmentation_tensor[AUG_TENSOR_COLOR_QUANT] = float(quant_div) / 5.0

                    pass
                elif 'gaussian_blur' in aug:
                    # Gaussian Blur
                    kernel = random.randint(1, 3) * 3
                    image = cv2.GaussianBlur(image, (kernel, kernel), 3)
                    augmentation_tensor[AUG_TENSOR_BLUR] = float(kernel) / 9
                elif 'median_blur' in aug:
                    # Median Blur
                    kernel = random.randint(1, 3) * 3
                    image = cv2.medianBlur(image, kernel)
                    augmentation_tensor[AUG_TENSOR_BLUR] = float(kernel) / 9
                elif 'motion_blur' in aug:
                    # Motion blur
                    intensity = random.randrange(1, 9)
                    image = self.motion_blur(image, intensity, random.randint(0, 360))
                    augmentation_tensor[AUG_TENSOR_BLUR] = intensity / 9
                elif 'smooth_blur' in aug:
                    # Smooth blur
                    kernel = random.randint(1, 3) * 3
                    image = cv2.blur(image, ksize=kernel)
                    augmentation_tensor[AUG_TENSOR_BLUR] = kernel / 9
                elif 'block_noise' in aug:
                    # Block noise
                    noise_intensity = random.randint(3, 10)
                    image += np.random.randn()
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
                    # Noise
                    pass
                elif 'jpeg' in aug:
                    # JPEG compression
                    pass
                elif 'saturation' in aug:
                    # Lightening / saturation
                    pass

        return corrupted_imgs