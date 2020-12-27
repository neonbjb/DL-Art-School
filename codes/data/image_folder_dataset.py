import glob
import itertools
import random

import cv2
import kornia
import numpy as np
import torch
import os

from data import util
# Builds a dataset created from a simple folder containing a list of training/test/validation images.
from data.image_corruptor import ImageCorruptor
from data.image_label_parser import VsNetImageLabeler


class ImageFolderDataset:
    def __init__(self, opt):
        self.opt = opt
        self.corruptor = ImageCorruptor(opt)
        self.target_hq_size = opt['target_size'] if 'target_size' in opt.keys() else None
        self.multiple = opt['force_multiple'] if 'force_multiple' in opt.keys() else 1
        self.scale = opt['scale']
        self.paths = opt['paths']
        self.corrupt_before_downsize = opt['corrupt_before_downsize'] if 'corrupt_before_downsize' in opt.keys() else False
        self.fetch_alt_image = opt['fetch_alt_image']  # If specified, this dataset will attempt to find a second image
                                                       # from the same video source. Search for 'fetch_alt_image' for more info.
        self.skip_lq = opt['skip_lq']
        self.disable_flip = opt['disable_flip']
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
        local_scale = self.scale
        if self.corrupt_before_downsize:
            # You can downsize to a specified scale, then corrupt, then continue the downsize further using this option.
            if 'corrupt_before_downsize_factor' in self.opt.keys():
                special_factor = self.opt['corrupt_before_downsize_factor']
                hs = [cv2.resize(h_, (h // special_factor, w // special_factor), interpolation=cv2.INTER_AREA) for h_ in hs]
                local_scale = local_scale // special_factor
            else:
                hs = [h.copy() for h in hs]
            hs = self.corruptor.corrupt_images(hs)
        for hq in hs:
            h, w, _ = hq.shape
            ls.append(cv2.resize(hq, (h // local_scale, w // local_scale), interpolation=cv2.INTER_AREA))
        # Corrupt the LQ image (only in eval mode)
        if not self.corrupt_before_downsize:
            ls = self.corruptor.corrupt_images(ls)
        return ls

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        hq = util.read_img(None, self.image_paths[item], rgb=True)
        if not self.disable_flip and random.random() < .5:
            hq = hq[:, ::-1, :]

        if self.labeler:
            assert hq.shape[0] == hq.shape[1]  # This just has not been accomodated yet.
            dim = hq.shape[0]

        hs = self.resize_hq([hq])
        if not self.skip_lq:
            ls = self.synthesize_lq(hs)

        # Convert to torch tensor
        hq = torch.from_numpy(np.ascontiguousarray(np.transpose(hs[0], (2, 0, 1)))).float()
        if not self.skip_lq:
            lq = torch.from_numpy(np.ascontiguousarray(np.transpose(ls[0], (2, 0, 1)))).float()

        out_dict = {'hq': hq, 'LQ_path': self.image_paths[item], 'HQ_path': self.image_paths[item]}
        if not self.skip_lq:
            out_dict['lq'] = lq

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
                    alt_hq = self.resize_hq([alt_hq])
                    alt_hq = torch.from_numpy(np.ascontiguousarray(np.transpose(alt_hq[0], (2, 0, 1)))).float()
                    if not self.skip_lq:
                        alt_lq = self.synthesize_lq(alt_hq)
                        alt_lq = torch.from_numpy(np.ascontiguousarray(np.transpose(alt_lq[0], (2, 0, 1)))).float()
                except:
                    alt_hq = hq
                    if not self.skip_lq:
                        alt_lq = lq
            else:
                alt_hq = hq
                if not self.skip_lq:
                    alt_lq = lq
            out_dict['alt_hq'] = alt_hq
            if not self.skip_lq:
                out_dict['alt_lq'] = alt_lq

        if self.labeler:
            base_file = self.image_paths[item].replace(self.paths[0], "")
            while base_file.startswith("\\"):
                base_file = base_file[1:]
            assert dim % hq.shape[1] == 0
            lbls, lbl_masks, lblstrings = self.labeler.get_labels_as_tensor(hq, base_file, dim // hq.shape[1])
            out_dict['labels'] = lbls
            out_dict['labels_mask'] = lbl_masks
            out_dict['label_strings'] = lblstrings
        return out_dict

if __name__ == '__main__':
    opt = {
        'name': 'amalgam',
        'paths': ['F:\\4k6k\\datasets\\images\\youtube\\4k_quote_unquote\\images'],
        'weights': [1],
        'target_size': 256,
        'force_multiple': 32,
        'scale': 2,
        'fixed_corruptions': ['jpeg-broad', 'gaussian_blur'],
        'random_corruptions': ['noise-5', 'none'],
        'num_corrupts_per_image': 1,
        'corrupt_before_downsize': True,
        'fetch_alt_image': True,
        #'labeler': {
        #    'type': 'patch_labels',
        #    'label_file': 'F:\\4k6k\\datasets\\ns_images\\512_unsupervised\\categories_new.json'
        #}
    }

    ds = ImageFolderDataset(opt)
    import os
    os.makedirs("debug", exist_ok=True)
    for i in range(0, len(ds)):
        o = ds[random.randint(0, len(ds)-1)]
        hq = o['hq']
        #masked = (o['labels_mask'] * .5 + .5) * hq
        import torchvision
        torchvision.utils.save_image(hq.unsqueeze(0), "debug/%i_hq.png" % (i,))
        torchvision.utils.save_image(o['alt_hq'].unsqueeze(0), "debug/%i_hq_alt.png" % (i,))
        #if len(o['labels'].unique()) > 1:
        #    randlbl = np.random.choice(o['labels'].unique()[1:])
        #    moremask = hq * ((1*(o['labels'] == randlbl))*.5+.5)
        #    torchvision.utils.save_image(moremask.unsqueeze(0), "debug/%i_%s.png" % (i, o['label_strings'][randlbl]))