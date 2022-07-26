import functools
import random

import cv2
import numpy as np
import torch
import os

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import tqdm

from data import util
# Builds a dataset created from a simple folder containing a list of training/test/validation images.
from data.images.image_corruptor import ImageCorruptor, kornia_color_jitter_numpy
from data.images.image_label_parser import VsNetImageLabeler
from utils.util import opt_get


def ndarray_center_crop(crop, img):
    y, x, c = img.shape
    startx = x // 2 - crop // 2
    starty = y // 2 - crop // 2
    return img[starty:starty + crop, startx:startx + crop, :]


class ImageFolderDataset:
    def __init__(self, opt):
        self.opt = opt
        self.corruptor = ImageCorruptor(opt)
        if 'center_crop_hq_sz' in opt.keys():
            self.center_crop = functools.partial(ndarray_center_crop, opt['center_crop_hq_sz'])
        self.target_hq_size = opt['target_size'] if 'target_size' in opt.keys() else None
        self.multiple = opt['force_multiple'] if 'force_multiple' in opt.keys() else 1
        self.scale = opt['scale']
        self.paths = opt['paths']
        self.corrupt_before_downsize = opt['corrupt_before_downsize'] if 'corrupt_before_downsize' in opt.keys() else False
        self.fetch_alt_image = opt['fetch_alt_image']  # If specified, this dataset will attempt to find a second image
                                                       # from the same video source. Search for 'fetch_alt_image' for more info.
        self.fetch_alt_tiled_image = opt['fetch_alt_tiled_image']  # If specified, this dataset will attempt to find anoter tile from the same source image
                                                                   #  Search for 'fetch_alt_tiled_image' for more info.
        assert not (self.fetch_alt_image and self.fetch_alt_tiled_image)  # These are mutually exclusive.
        self.skip_lq = opt_get(opt, ['skip_lq'], False)
        self.disable_flip = opt_get(opt, ['disable_flip'], False)
        self.rgb_n1_to_1 = opt_get(opt, ['rgb_n1_to_1'], False)
        self.force_square = opt_get(opt, ['force_square'], True)
        self.fixed_parameters = {k: torch.tensor(v) for k, v in opt_get(opt, ['fixed_parameters'], {}).items()}
        self.all_image_color_jitter = opt_get(opt, ['all_image_color_jitter'], 0)
        if 'normalize' in opt.keys():
            if opt['normalize'] == 'stylegan2_norm':
                self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            elif opt['normalize'] == 'imagenet':
                self.normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            else:
                raise Exception('Unsupported normalize')
        else:
            self.normalize = None
        if self.target_hq_size is not None:
            assert (self.target_hq_size // self.scale) % self.multiple == 0  # If we dont throw here, we get some really obscure errors.
        if not isinstance(self.paths, list):
            self.paths = [self.paths]
            self.weights = [1]
        else:
            self.weights = opt['weights']

        if 'labeler' in opt.keys():
            if opt['labeler']['type'] == 'patch_labels':
                self.labeler = VsNetImageLabeler(opt['labeler']['label_file'])
            assert len(self.paths) == 1   # Only a single base-path is supported for labeled images.
            self.image_paths = self.labeler.get_labeled_paths(self.paths[0])
        else:
            self.labeler = None

            # Just scan the given directory for images of standard types.
            supported_types = ['jpg', 'jpeg', 'png', 'gif']
            self.image_paths = []
            for path, weight in zip(self.paths, self.weights):
                cache_path = os.path.join(path, 'cache.pth')
                if os.path.exists(cache_path):
                    imgs = torch.load(cache_path)
                else:
                    print("Building image folder cache, this can take some time for large datasets..")
                    imgs = util.find_files_of_type('img', path)[0]
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
        local_scale = self.scale
        if self.corrupt_before_downsize:
            # You can downsize to a specified scale, then corrupt, then continue the downsize further using this option.
            if 'corrupt_before_downsize_factor' in self.opt.keys():
                special_factor = self.opt['corrupt_before_downsize_factor']
                hs = [cv2.resize(h_, (h // special_factor, w // special_factor), interpolation=cv2.INTER_AREA) for h_ in hs]
                local_scale = local_scale // special_factor
            else:
                hs = [h.copy() for h in hs]
            hs, ent = self.corruptor.corrupt_images(hs, return_entropy=True)
        for hq in hs:
            h, w, _ = hq.shape
            ls.append(cv2.resize(hq, (h // local_scale, w // local_scale), interpolation=cv2.INTER_AREA))
        # Corrupt the LQ image (only in eval mode)
        if not self.corrupt_before_downsize:
            ls, ent = self.corruptor.corrupt_images(ls, return_entropy=True)
        return ls, ent

    def reset_random(self):
        self.corruptor.reset_random()

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        hq = util.read_img(None, self.image_paths[item], rgb=True)
        if hasattr(self, 'center_crop'):
            hq = self.center_crop(hq)
        if not self.disable_flip and random.random() < .5:
            hq = hq[:, ::-1, :]

        if self.force_square:
            h, w, _ = hq.shape
            dim = min(h, w)
            hq = hq[(h - dim) // 2:dim + (h - dim) // 2, (w - dim) // 2:dim + (w - dim) // 2, :]

        # Perform color jittering on the HQ image if specified. The given value should be between [0,1].
        if self.all_image_color_jitter > 0:
            hq = kornia_color_jitter_numpy(hq, self.all_image_color_jitter)

        if self.labeler:
            assert hq.shape[0] == hq.shape[1]  # This just has not been accomodated yet.
            dim = hq.shape[0]

        hs = self.resize_hq([hq])
        if not self.skip_lq:
            for_lq = [hs[0]]

        # Convert to torch tensor
        hq = torch.from_numpy(np.ascontiguousarray(np.transpose(hs[0], (2, 0, 1)))).float()

        out_dict = {'hq': hq, 'LQ_path': self.image_paths[item], 'HQ_path': self.image_paths[item], 'has_alt': False}

        if self.fetch_alt_image:
            # This works by assuming a specific filename structure as would produced by ffmpeg. ex:
            # 'Candied Walnutsxjktqhr_SYc.webm_00000478.jpg` and
            # 'Candied Walnutsxjktqhr_SYc.webm_00000479.jpg` and
            # 'Candied Walnutsxjktqhr_SYc.webm_00000480.jpg`
            # The basic format is `<anything>%08d.<extension>`. This logic parses off that 8 digit number. If it is
            # not found, the 'alt_image' returned is just the current image. If it is found, the algorithm searches for
            # an image one number higher. If it is found - it is returned in the 'alt_hq' and 'alt_lq' keys, else the
            # current image is put in those keys.

            imname_parts = self.image_paths[item]
            while '.jpg.jpg' in imname_parts:
                imname_parts = imname_parts.replace(".jpg.jpg", ".jpg")  # Hack workaround to my own bug.
            imname_parts = imname_parts.split('.')
            if len(imname_parts) >= 2 and len(imname_parts[-2]) > 8:
                try:
                    imnumber = int(imname_parts[-2][-8:])
                    # When we're dealing with images in the 1M range, it's straight up faster to attempt to just open
                    # the file rather than searching the path list. Let the exception handler below do its work.
                    next_img = self.image_paths[item].replace(str(imnumber), str(imnumber+1))
                    alt_hq = util.read_img(None, next_img, rgb=True)
                    alt_hs = self.resize_hq([alt_hq])
                    alt_hq = torch.from_numpy(np.ascontiguousarray(np.transpose(alt_hs[0], (2, 0, 1)))).float()
                    out_dict['has_alt'] = True
                    if not self.skip_lq:
                        for_lq.append(alt_hs[0])
                except:
                    alt_hq = hq
                    if not self.skip_lq:
                        for_lq.append(hs[0])
            else:
                alt_hq = hq
                if not self.skip_lq:
                    for_lq.append(hs[0])
            out_dict['alt_hq'] = alt_hq
            
        if self.fetch_alt_tiled_image:
            # This assumes the output format generated by the tiled image generation scripts included with DLAS. Specifically,
            # all image read by this dataset are assumed to be in subfolders with other tiles from the same source image. When
            # this option is set, another random image from the same folder is selected and returned as the alt image.
            sel_path = os.path.dirname(self.image_paths[item])
            other_images = os.listdir(sel_path)
            # Assume that the directory contains at least <image>, <ref.jpg>, <centers.pt>
            try:
                if len(other_images) <= 3:
                    alt_hq = hq  # This is a fallback in case an alt image can't be found.
                else:
                    random.shuffle(other_images)
                    for oi in other_images:
                        if oi == os.path.basename(self.image_paths[item]) or 'ref.' in oi or 'centers.pt' in oi:
                            continue
                        alt_hq = util.read_img(None, os.path.join(sel_path, oi), rgb=True)
                        alt_hs = self.resize_hq([alt_hq])
                        alt_hq = torch.from_numpy(np.ascontiguousarray(np.transpose(alt_hs[0], (2, 0, 1)))).float()
            except:
                alt_hq = hq
                print(f"Error with {self.image_paths[item]}")
            out_dict['has_alt'] = True
            out_dict['alt_hq'] = alt_hq


        if not self.skip_lq:
            lqs, ent = self.synthesize_lq(for_lq)
            ls = lqs[0]
            out_dict['lq'] = torch.from_numpy(np.ascontiguousarray(np.transpose(ls, (2, 0, 1)))).float()
            out_dict['corruption_entropy'] = torch.tensor(ent)
            if len(lqs) > 1:
                alt_lq = lqs[1]
                out_dict['alt_lq'] = torch.from_numpy(np.ascontiguousarray(np.transpose(alt_lq, (2, 0, 1)))).float()


        if self.labeler:
            base_file = self.image_paths[item].replace(self.paths[0], "")
            while base_file.startswith("\\"):
                base_file = base_file[1:]
            assert dim % hq.shape[1] == 0
            lbls, lbl_masks, lblstrings = self.labeler.get_labels_as_tensor(hq, base_file, dim // hq.shape[1])
            out_dict['labels'] = lbls
            out_dict['labels_mask'] = lbl_masks
            out_dict['label_strings'] = lblstrings

        for k, v in out_dict.items():
            if isinstance(v, torch.Tensor) and len(v.shape) == 3:
                if self.normalize:
                    v = self.normalize(v)
                if self.rgb_n1_to_1:
                    v = v * 2 - 1
                out_dict[k] = v

        out_dict.update(self.fixed_parameters)
        return out_dict

if __name__ == '__main__':
    opt = {
        'name': 'amalgam',
        'paths': ['E:\\4k6k\\datasets\\ns_images\\imagesets\\256_only_humans_masked'],
        'weights': [1],
        'target_size': 256,
        'scale': 1,
        'corrupt_before_downsize': True,
        'fetch_alt_image': False,
        'fetch_alt_tiled_image': True,
        'disable_flip': True,
        'fixed_corruptions': ['lq_resampling', 'jpeg-medium', 'gaussian_blur', 'noise', 'color_jitter'],
        'num_corrupts_per_image': 0,
        'corruption_blur_scale': 1,
        'all_image_color_jitter': .1,
    }

    ds = DataLoader(ImageFolderDataset(opt), shuffle=True, num_workers=0, batch_size=64)
    import os
    output_path = 'F:\\tmp'
    os.makedirs(output_path, exist_ok=True)
    res = []
    for i, d in tqdm(enumerate(ds)):
        '''
        x = d['hq']
        b,c,h,w = x.shape
        x_c = x.view(c*b, h, w)
        x_c = torch.view_as_real(torch.fft.rfft(x_c))
        # Log-normalize spectrogram
        x_c = (x_c.abs() ** 2).clip(min=1e-8, max=1e16)
        x_c = torch.log(x_c)
        res.append(x_c)
        if i % 100 == 99:
            stacked = torch.cat(res, dim=0)
            print(stacked.mean(dim=[0,1,2]), stacked.std(dim=[0,1,2]))
        '''

        for k, v in d.items():
            if isinstance(v, torch.Tensor) and len(v.shape) >= 3:
                os.makedirs(f'{output_path}\\{k}', exist_ok=True)
                torchvision.utils.save_image(v, f'{output_path}\\{k}\\{i}.png')
        if i >= 200000:
            break
