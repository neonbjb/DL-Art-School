import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
from PIL import Image, ImageOps
from io import BytesIO
import torchvision.transforms.functional as F


# Reads full-quality images and pulls tiles from them. Also extracts LR renderings of the full image with cues as to
# where those tiles came from.
class FullImageDataset(data.Dataset):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    If only GT images are provided, generate LQ images on-the-fly.
    """
    def get_lq_path(self, i):
        which_lq = random.randint(0, len(self.paths_LQ)-1)
        return self.paths_LQ[which_lq][i % len(self.paths_LQ[which_lq])]

    def __init__(self, opt):
        super(FullImageDataset, self).__init__()
        self.opt = opt
        self.data_type = 'img'
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None
        self.force_multiple = self.opt['force_multiple'] if 'force_multiple' in self.opt.keys() else 1

        self.paths_GT, self.sizes_GT = util.find_files_of_type(self.data_type, opt['dataroot_GT'], opt['dataroot_GT_weights'])
        if 'dataroot_LQ' in opt.keys():
            self.paths_LQ = []
            if isinstance(opt['dataroot_LQ'], list):
                # Multiple LQ data sources can be given, in case there are multiple ways of corrupting a source image and
                # we want the model to learn them all.
                for dr_lq in opt['dataroot_LQ']:
                    lq_path, self.sizes_LQ = util.find_files_of_type(self.data_type, dr_lq)
                    self.paths_LQ.append(lq_path)
            else:
                lq_path, self.sizes_LQ = util.find_files_of_type(self.data_type, opt['dataroot_LQ'])
                self.paths_LQ.append(lq_path)

        assert self.paths_GT, 'Error: GT path is empty.'
        self.random_scale_list = [1]

    def motion_blur(self, image, size, angle):
        k = np.zeros((size, size), dtype=np.float32)
        k[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
        k = k * (1.0 / np.sum(k))
        return cv2.filter2D(image, -1, k)

    # Selects the smallest dimension from the image and crops it randomly so the other dimension matches. The cropping
    # offset from center is chosen on a normal probability curve.
    def get_square_image(self, image):
        h, w, _ = image.shape
        if h == w:
            return image
        offset = max(min(np.random.normal(scale=.3), 1.0), -1.0)
        if h > w:
            diff = h - w
            center = diff // 2
            top = int(center + offset * (center - 2))
            return image[top:top+w, :, :]
        else:
            diff = w - h
            center = diff // 2
            left = int(center + offset * (center - 2))
            return image[:, left:left+h, :]

    def pick_along_range(self, sz, r, dev):
        margin_sz = sz - r
        margin_center = margin_sz // 2
        return min(max(int(min(np.random.normal(scale=dev), 1.0) * margin_sz + margin_center), 0), margin_sz)

    def resize_point(self, point, orig_dim, new_dim):
        oh, ow = orig_dim
        nh, nw = new_dim
        dh, dw = float(nh) / float(oh), float(nw) / float(ow)
        point[0] = int(dh * float(point[0]))
        point[1] = int(dw * float(point[1]))
        return point

    # - Randomly extracts a square from image and resizes it to opt['target_size'].
    # - Fills a mask with zeros, then places 1's where the square was extracted from. Resizes this mask and the source
    #   image to the target_size and returns that too.
    # Notes:
    # - When extracting a square, the size of the square is randomly distributed [target_size, source_size] along a
    #   half-normal distribution, biasing towards the target_size.
    # - A biased normal distribution is also used to bias the tile selection towards the center of the source image.
    def pull_tile(self, image, lq=False):
        if lq:
            target_sz = self.opt['min_tile_size'] // self.opt['scale']
        else:
            target_sz = self.opt['min_tile_size']
        h, w, _ = image.shape
        possible_sizes_above_target = h - target_sz
        if 'fixed_size' in self.opt.keys() and self.opt['fixed_size']:
            square_size = target_sz
        else:
            tile_expansion_dev = self.opt['tile_scale_normal_stddev'] if 'tile_scale_normal_stddev' in self.opt.keys() else .17
            square_size = int(target_sz + possible_sizes_above_target * min(np.abs(np.random.normal(scale=tile_expansion_dev)), 1.0))

        # Pick the left,top coords to draw the patch from
        left = self.pick_along_range(w, square_size, .3)
        top = self.pick_along_range(w, square_size, .3)

        mask = np.zeros((h, w, 1), dtype=image.dtype)
        mask[top:top+square_size, left:left+square_size] = 1
        patch = image[top:top+square_size, left:left+square_size, :]
        center = torch.tensor([top + square_size // 2, left + square_size // 2], dtype=torch.long)

        patch = cv2.resize(patch, (target_sz, target_sz), interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, (target_sz, target_sz), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (target_sz, target_sz), interpolation=cv2.INTER_LINEAR)
        center = self.resize_point(center, (h, w), image.shape[:2])

        return patch, image, mask, center

    def augment_tile(self, img_GT, img_LQ, strength=1):
        scale = self.opt['scale']
        GT_size = self.opt['target_size']

        H, W, _ = img_GT.shape
        assert H >= GT_size and W >= GT_size

        LQ_size = GT_size // scale
        img_LQ = cv2.resize(img_LQ, (LQ_size, LQ_size), interpolation=cv2.INTER_LINEAR)
        img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)

        if self.opt['use_blurring']:
            # Pick randomly between gaussian, motion, or no blur.
            blur_det = random.randint(0, 100)
            blur_magnitude = 3 if 'blur_magnitude' not in self.opt.keys() else self.opt['blur_magnitude']
            blur_magnitude = max(1, int(blur_magnitude*strength))
            if blur_det < 40:
                blur_sig = int(random.randrange(0, int(blur_magnitude)))
                img_LQ = cv2.GaussianBlur(img_LQ, (blur_magnitude, blur_magnitude), blur_sig)
            elif blur_det < 70:
                img_LQ = self.motion_blur(img_LQ, random.randrange(1, int(blur_magnitude) * 3), random.randint(0, 360))

        return img_GT, img_LQ

    # Converts img_LQ to PIL and performs JPG compression corruptions and grayscale on the image, then returns it.
    def pil_augment(self, img_LQ, strength=1):
        img_LQ = (img_LQ * 255).astype(np.uint8)
        img_LQ = Image.fromarray(img_LQ)
        if self.opt['use_compression_artifacts'] and random.random() > .25:
            sub_lo = 90 * strength
            sub_hi = 30 * strength
            qf = random.randrange(100 - sub_lo, 100 - sub_hi)
            corruption_buffer = BytesIO()
            img_LQ.save(corruption_buffer, "JPEG", quality=qf, optimice=True)
            corruption_buffer.seek(0)
            img_LQ = Image.open(corruption_buffer)

        if 'grayscale' in self.opt.keys() and self.opt['grayscale']:
            img_LQ = ImageOps.grayscale(img_LQ).convert('RGB')

        return img_LQ

    def perform_random_hr_augment(self, image, aug_code=None, augmentations=1):
        if aug_code is None:
            aug_code = [random.randint(0, 10) for _ in range(augmentations)]
        else:
            assert augmentations == 1
            aug_code = [aug_code]
        if 0 in aug_code:
            # Color quantization
            pass
        elif 1 in aug_code:
            # Gaussian Blur (point or motion)
            blur_magnitude = 3
            blur_sig = int(random.randrange(0, int(blur_magnitude)))
            image = cv2.GaussianBlur(image, (blur_magnitude, blur_magnitude), blur_sig)
        elif 2 in aug_code:
            # Median Blur
            image = cv2.medianBlur(image, 3)
        elif 3 in aug_code:
            # Motion blur
            image = self.motion_blur(image, random.randrange(1, 9), random.randint(0, 360))
        elif 4 in aug_code:
            # Smooth blur
            image = cv2.blur(image, ksize=3)
        elif 5 in aug_code:
            # Block noise
            pass
        elif 6 in aug_code:
            # Bicubic LR->HR
            pass
        elif 7 in aug_code:
            # Linear compression distortion
            pass
        elif 8 in aug_code:
            # Interlacing distortion
            pass
        elif 9 in aug_code:
            # Chromatic aberration
            pass
        elif 10 in aug_code:
            # Noise
            pass
        elif 11 in aug_code:
            # JPEG compression
            pass
        elif 12 in aug_code:
            # Lightening / saturation
            pass
        return image

    def __getitem__(self, index):
        scale = self.opt['scale']

        # get full size image
        full_path = self.paths_GT[index % len(self.paths_GT)]
        LQ_path = full_path
        img_full = util.read_img(None, full_path, None)
        img_full = util.channel_convert(img_full.shape[2], 'RGB', [img_full])[0]
        if self.opt['phase'] == 'train':
            img_full = util.augment([img_full], self.opt['use_flip'], self.opt['use_rot'])[0]
            img_full = self.get_square_image(img_full)
            img_GT, gt_fullsize_ref, gt_mask, gt_center = self.pull_tile(img_full)
        else:
            img_GT, gt_fullsize_ref = img_full, img_full
            gt_mask = np.ones(img_full.shape[:2], dtype=gt_fullsize_ref.dtype)
            gt_center = torch.tensor([img_full.shape[0] // 2, img_full.shape[1] // 2], dtype=torch.long)
        orig_gt_dim = gt_fullsize_ref.shape[:2]

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.get_lq_path(index)
            img_lq_full = util.read_img(None, LQ_path, None)
            if self.opt['phase'] == 'train':
                img_lq_full = util.augment([img_lq_full], self.opt['use_flip'], self.opt['use_rot'])[0]
                img_lq_full = self.get_square_image(img_lq_full)
                img_LQ, lq_fullsize_ref, lq_mask, lq_center = self.pull_tile(img_lq_full, lq=True)
            else:
                img_LQ, lq_fullsize_ref = img_lq_full, img_lq_full
                lq_mask = np.ones(img_lq_full.shape[:2], dtype=lq_fullsize_ref.dtype)
                lq_center = torch.tensor([img_lq_full.shape[0] // 2, img_lq_full.shape[1] // 2], dtype=torch.long)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                GT_size = self.opt['target_size']
                random_scale = random.choice(self.random_scale_list)
                if len(img_GT.shape) == 2:
                    print("ERRAR:")
                    print(img_GT.shape)
                    print(full_path)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(img_GT, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape

            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            lq_fullsize_ref = util.imresize_np(gt_fullsize_ref, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)
            lq_mask, lq_center = gt_mask, self.resize_point(gt_center.clone(), orig_gt_dim, lq_fullsize_ref.shape[:2])
        orig_lq_dim = lq_fullsize_ref.shape[:2]

        # Enforce force_resize constraints via clipping.
        h, w, _ = img_LQ.shape
        if h % self.force_multiple != 0 or w % self.force_multiple != 0:
            h, w = (h - h % self.force_multiple), (w - w % self.force_multiple)
            img_LQ = img_LQ[:h, :w, :]
            lq_fullsize_ref = lq_fullsize_ref[:h, :w, :]
            h *= scale
            w *= scale
            img_GT = img_GT[:h, :w]
            gt_fullsize_ref = gt_fullsize_ref[:h, :w, :]

        if self.opt['phase'] == 'train':
            img_GT, img_LQ = self.augment_tile(img_GT, img_LQ)
            gt_fullsize_ref, lq_fullsize_ref = self.augment_tile(gt_fullsize_ref, lq_fullsize_ref, strength=.2)

        # Scale masks.
        lq_mask = cv2.resize(lq_mask, (lq_fullsize_ref.shape[1], lq_fullsize_ref.shape[0]), interpolation=cv2.INTER_LINEAR)
        gt_mask = cv2.resize(gt_mask, (gt_fullsize_ref.shape[1], gt_fullsize_ref.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Scale center coords
        lq_center = self.resize_point(lq_center, orig_lq_dim, lq_fullsize_ref.shape[:2])
        gt_center = self.resize_point(gt_center, orig_gt_dim, gt_fullsize_ref.shape[:2])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = cv2.cvtColor(img_GT, cv2.COLOR_BGR2RGB)
            img_LQ = cv2.cvtColor(img_LQ, cv2.COLOR_BGR2RGB)
            lq_fullsize_ref = cv2.cvtColor(lq_fullsize_ref, cv2.COLOR_BGR2RGB)
            gt_fullsize_ref = cv2.cvtColor(gt_fullsize_ref, cv2.COLOR_BGR2RGB)

        # LQ needs to go to a PIL image to perform the compression-artifact transformation.
        #if self.opt['phase'] == 'train':
            #img_LQ = self.pil_augment(img_LQ)
            #lq_fullsize_ref = self.pil_augment(lq_fullsize_ref, strength=.2)

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        gt_fullsize_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(gt_fullsize_ref, (2, 0, 1)))).float()
        img_LQ = F.to_tensor(img_LQ)
        lq_fullsize_ref = F.to_tensor(lq_fullsize_ref)
        lq_mask = torch.from_numpy(np.ascontiguousarray(lq_mask)).unsqueeze(dim=0)
        gt_mask = torch.from_numpy(np.ascontiguousarray(gt_mask)).unsqueeze(dim=0)

        if 'lq_noise' in self.opt.keys():
            lq_noise = torch.randn_like(img_LQ) * self.opt['lq_noise'] / 255
            img_LQ += lq_noise
            lq_fullsize_ref += lq_noise

        # Apply the masks to the full images.
        gt_fullsize_ref = torch.cat([gt_fullsize_ref, gt_mask], dim=0)
        lq_fullsize_ref = torch.cat([lq_fullsize_ref, lq_mask], dim=0)

        d = {'lq': img_LQ, 'hq': img_GT, 'gt_fullsize_ref': gt_fullsize_ref, 'lq_fullsize_ref': lq_fullsize_ref,
             'lq_center': lq_center, 'gt_center': gt_center,
             'LQ_path': LQ_path, 'HQ_path': full_path}
        return d

    def __len__(self):
        return len(self.paths_GT)

if __name__ == '__main__':
    '''
    opt = {
        'name': 'amalgam',
        'dataroot_GT': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\images'],
        'dataroot_GT_weights': [1],
        'use_flip': True,
        'use_compression_artifacts': True,
        'use_blurring': True,
        'use_rot': True,
        'lq_noise': 5,
        'target_size': 128,
        'min_tile_size': 256,
        'scale': 2,
        'phase': 'train'
    }
    '''
    opt = {
        'name': 'amalgam',
        'dataroot_GT': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\images'],
        'dataroot_GT_weights': [1],
        'force_multiple': 32,
        'scale': 2,
        'phase': 'test'
    }

    ds = FullImageDataset(opt)
    import os
    os.makedirs("debug", exist_ok=True)
    for i in range(300, len(ds)):
        print(i)
        o = ds[i]
        for k, v in o.items():
            if 'path' not in k:
                #if 'full' in k:
                    #masked = v[:3, :, :] * v[3]
                    #torchvision.utils.save_image(masked.unsqueeze(0), "debug/%i_%s_masked.png" % (i, k))
                    #v = v[:3, :, :]
                #import torchvision
                #torchvision.utils.save_image(v.unsqueeze(0), "debug/%i_%s.png" % (i, k))
                pass
