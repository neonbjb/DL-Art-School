import torch
from torch.utils import data
from data.images.image_corruptor import ImageCorruptor
from data.images.chunk_with_reference import ChunkWithReference
import os
import cv2
import numpy as np

# Class whose purpose is to hold as much logic as can possibly be shared between datasets that operate on raw image
# data and nothing else (which also have a very specific directory structure being used, as dictated by
# ChunkWithReference).
class BaseUnsupervisedImageDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.corruptor = ImageCorruptor(opt)
        self.target_hq_size = opt['target_size'] if 'target_size' in opt.keys() else None
        self.multiple = opt['force_multiple'] if 'force_multiple' in opt.keys() else 1
        self.for_eval = opt['eval'] if 'eval' in opt.keys() else False
        self.scale = opt['scale'] if not self.for_eval else 1
        self.paths = opt['paths']
        self.corrupt_before_downsize = opt['corrupt_before_downsize'] if 'corrupt_before_downsize' in opt.keys() else False
        assert (self.target_hq_size // self.scale) % self.multiple == 0  # If we dont throw here, we get some really obscure errors.
        if not isinstance(self.paths, list):
            self.paths = [self.paths]
            self.weights = [1]
        else:
            self.weights = opt['weights']

        # See if there is a cached directory listing and use that rather than re-scanning everything. This will greatly
        # reduce startup costs.
        self.chunks = []
        for path, weight in zip(self.paths, self.weights):
            cache_path = os.path.join(path, 'cache.pth')
            if os.path.exists(cache_path):
                chunks = torch.load(cache_path)
            else:
                print("Building chunk cache, this can take some time for large datasets..")
                chunks = [ChunkWithReference(opt, d) for d in sorted(os.scandir(path), key=lambda e: e.name) if d.is_dir()]
                # Prune out chunks that have no images
                res = []
                for c in chunks:
                    if len(c) != 0:
                        res.append(c)
                chunks = res
                # Save to a cache.
                torch.save(chunks, cache_path)
            for w in range(weight):
                self.chunks.extend(chunks)

        # Indexing this dataset is tricky. Aid it by having a list of starting indices for each chunk.
        start = 0
        self.starting_indices = []
        for c in self.chunks:
            self.starting_indices.append(start)
            start += len(c)
        self.len = start

    def get_paths(self):
        paths = []
        for c in self.chunks:
            paths.extend(c.tiles)
        return paths
        
    # Utility method for translating a point when the dimensions of an image change.
    def resize_point(self, point, orig_dim, new_dim):
        oh, ow = orig_dim
        nh, nw = new_dim
        dh, dw = float(nh) / float(oh), float(nw) / float(ow)
        point = int(dh * float(point[0])), int(dw * float(point[1]))
        return point

    # Given an HQ square of arbitrary size, resizes it to specifications from opt.
    def resize_hq(self, imgs_hq, refs_hq, masks_hq, centers_hq):
        # Enforce size constraints
        h, w, _ = imgs_hq[0].shape
        if self.target_hq_size is not None and self.target_hq_size != h:
            hqs_adjusted, hq_refs_adjusted, hq_masks_adjusted, hq_centers_adjusted = [], [], [], []
            for hq, hq_ref, hq_mask, hq_center in zip(imgs_hq, refs_hq, masks_hq, centers_hq):
                # It is assumed that the target size is a square.
                target_size = (self.target_hq_size, self.target_hq_size)
                hqs_adjusted.append(cv2.resize(hq, target_size, interpolation=cv2.INTER_AREA))
                hq_refs_adjusted.append(cv2.resize(hq_ref, target_size, interpolation=cv2.INTER_AREA))
                hq_masks_adjusted.append(cv2.resize(hq_mask, target_size, interpolation=cv2.INTER_AREA))
                hq_centers_adjusted.append(self.resize_point(hq_center, (h, w), target_size))
            h, w = self.target_hq_size, self.target_hq_size
        else:
            hqs_adjusted, hq_refs_adjusted, hq_masks_adjusted, hq_centers_adjusted = imgs_hq, refs_hq, masks_hq, centers_hq
            hq_masks_adjusted = [m.squeeze(-1) for m in hq_masks_adjusted]  # This is done implicitly above..
        hq_multiple = self.multiple * self.scale   # Multiple must apply to LQ image.
        if h % hq_multiple != 0 or w % hq_multiple != 0:
            hqs_conformed, hq_refs_conformed, hq_masks_conformed, hq_centers_conformed = [], [], [], []
            for hq, hq_ref, hq_mask, hq_center in zip(hqs_adjusted, hq_refs_adjusted, hq_masks_adjusted, hq_centers_adjusted):
                h, w = (h - h % hq_multiple), (w - w % hq_multiple)
                hq_centers_conformed.append(self.resize_point(hq_center, hq.shape[:2], (h, w)))
                hqs_conformed.append(hq[:h, :w, :])
                hq_refs_conformed.append(hq_ref[:h, :w, :])
                hq_masks_conformed.append(hq_mask[:h, :w, :])
            return hqs_conformed, hq_refs_conformed, hq_masks_conformed, hq_centers_conformed
        return hqs_adjusted, hq_refs_adjusted, hq_masks_adjusted, hq_centers_adjusted

    def synthesize_lq(self, hs, hrefs, hmasks, hcenters):
        h, w, _ = hs[0].shape
        ls, lrs, lms, lcs = [], [], [], []
        if self.corrupt_before_downsize and not self.for_eval:
            hs = self.corruptor.corrupt_images(np.copy(hs))
        for hq, hq_ref, hq_mask, hq_center in zip(hs, hrefs, hmasks, hcenters):
            if self.for_eval:
                ls.append(hq)
                lrs.append(hq_ref)
                lms.append(hq_mask)
                lcs.append(hq_center)
            else:
                ls.append(cv2.resize(hq, (h // self.scale, w // self.scale), interpolation=cv2.INTER_AREA))
                lrs.append(cv2.resize(hq_ref, (h // self.scale, w // self.scale), interpolation=cv2.INTER_AREA))
                lms.append(cv2.resize(hq_mask, (h // self.scale, w // self.scale), interpolation=cv2.INTER_AREA))
                lcs.append(self.resize_point(hq_center, (h, w), ls[0].shape[:2]))
        # Corrupt the LQ image (only in eval mode)
        if not self.corrupt_before_downsize and not self.for_eval:
            ls = self.corruptor.corrupt_images(ls)
        return ls, lrs, lms, lcs

    def __len__(self):
        return self.len
