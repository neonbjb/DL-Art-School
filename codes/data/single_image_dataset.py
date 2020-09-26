from torch.utils import data
from data.chunk_with_reference import ChunkWithReference
from data.image_corruptor import ImageCorruptor
import os
from bisect import bisect_left
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as F


# Builds a dataset composed of a set of folders. Each folder represents a single high resolution image that has been
# chunked into patches of fixed size. A reference image is included as well as a list of center points for each patch.
class SingleImageDataset(data.Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.corruptor = ImageCorruptor(opt)
        self.target_hq_size = opt['target_size'] if 'target_size' in opt.keys() else None
        self.multiple = opt['force_multiple'] if 'force_multiple' in opt.keys() else 1
        self.for_eval = opt['eval'] if 'eval' in opt.keys() else False
        self.scale = opt['scale'] if not self.for_eval else 1
        self.paths = opt['paths']
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
                # Update the options.
                for c in chunks:
                    c.reload(opt)
            else:
                chunks = [ChunkWithReference(opt, d) for d in os.scandir(path) if d.is_dir()]
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

        # Indexing this dataset is tricky. Aid it by having a sorted list of starting indices for each chunk.
        start = 0
        self.starting_indices = []
        for c in chunks:
            self.starting_indices.append(start)
            start += len(c)
        self.len = start


    def resize_point(self, point, orig_dim, new_dim):
        oh, ow = orig_dim
        nh, nw = new_dim
        dh, dw = float(nh) / float(oh), float(nw) / float(ow)
        point = int(dh * float(point[0])), int(dw * float(point[1]))
        return point

    def __getitem__(self, item):
        chunk_ind = bisect_left(self.starting_indices, item)
        chunk_ind = chunk_ind if chunk_ind < len(self.starting_indices) and self.starting_indices[chunk_ind] == item else chunk_ind-1
        hq, hq_ref, hq_center, hq_mask, path = self.chunks[chunk_ind][item-self.starting_indices[chunk_ind]]

        # Enforce size constraints
        h, w, _ = hq.shape
        if self.target_hq_size is not None and self.target_hq_size != h:
            # It is assumed that the target size is a square.
            target_size = (self.target_hq_size, self.target_hq_size)
            hq = cv2.resize(hq, target_size, interpolation=cv2.INTER_LINEAR)
            hq_ref = cv2.resize(hq_ref, target_size, interpolation=cv2.INTER_LINEAR)
            hq_mask = cv2.resize(hq_mask, target_size, interpolation=cv2.INTER_LINEAR)
            hq_center = self.resize_point(hq_center, (h, w), target_size)
            h, w = self.target_hq_size, self.target_hq_size
        hq_multiple = self.multiple * self.scale   # Multiple must apply to LQ image.
        if h % hq_multiple != 0 or w % hq_multiple != 0:
            h, w = (h - h % hq_multiple), (w - w % hq_multiple)
            hq_center = self.resize_point(hq_center, hq.shape[:1], (h, w))
            hq = hq[:h, :w, :]
            hq_ref = hq_ref[:h, :w, :]
            hq_mask = hq_mask[:h, :w, :]

        # Synthesize the LQ image
        if self.for_eval:
            lq, lq_ref = hq, hq_ref
        else:
            lq = cv2.resize(hq, (h // self.scale, w // self.scale), interpolation=cv2.INTER_LINEAR)
            lq_ref = cv2.resize(hq_ref, (h // self.scale, w // self.scale), interpolation=cv2.INTER_LINEAR)
            lq_mask = cv2.resize(hq_mask, (h // self.scale, w // self.scale), interpolation=cv2.INTER_LINEAR)
            lq_center = self.resize_point(hq_center, (h, w), lq.shape[:2])

        # Corrupt the LQ image
        lq = self.corruptor.corrupt_images([lq])[0]

        # Convert to torch tensor
        hq = torch.from_numpy(np.ascontiguousarray(np.transpose(hq, (2, 0, 1)))).float()
        hq_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(hq_ref, (2, 0, 1)))).float()
        hq_mask = torch.from_numpy(np.ascontiguousarray(hq_mask)).unsqueeze(dim=0)
        hq_ref = torch.cat([hq_ref, hq_mask], dim=0)
        lq = torch.from_numpy(np.ascontiguousarray(np.transpose(lq, (2, 0, 1)))).float()
        lq_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(lq_ref, (2, 0, 1)))).float()
        lq_mask = torch.from_numpy(np.ascontiguousarray(lq_mask)).unsqueeze(dim=0)
        lq_ref = torch.cat([lq_ref, lq_mask], dim=0)

        return {'LQ': lq, 'GT': hq, 'gt_fullsize_ref': hq_ref, 'lq_fullsize_ref': lq_ref,
             'lq_center': torch.tensor(lq_center, dtype=torch.long), 'gt_center': torch.tensor(hq_center, dtype=torch.long),
             'LQ_path': path, 'GT_path': path}

    def __len__(self):
        return self.len


        self.corruptor = ImageCorruptor(opt)
        self.target_hq_size = opt['target_size'] if 'target_size' in opt.keys() else None
        self.multiple = opt['force_multiple'] if 'force_multiple' in opt.keys() else 1
        self.for_eval = opt['eval'] if 'eval' in opt.keys() else False
        self.scale = opt['scale'] if not self.for_eval else 1
        self.paths = opt['paths']
        if not isinstance(self.paths, list):
            self.paths = [self.paths]
            self.weights = [1]
        else:
            self.weights = opt['weights']
        for path, weight in zip(self.paths, self.weights):
            chunks = [ChunkWithReference(opt, d) for d in os.scandir(path) if d.is_dir()]
            for w in range(weight):
                self.chunks.extend(chunks)

if __name__ == '__main__':
    opt = {
        'name': 'amalgam',
        'paths': ['F:\\4k6k\\datasets\\images\\flickr\\testbed'],
        'weights': [1],
        'target_size': 128,
        'force_multiple': 32,
        'scale': 2,
        'eval': False,
        'fixed_corruptions': ['jpeg'],
        'random_corruptions': ['color_quantization', 'gaussian_blur', 'motion_blur', 'smooth_blur', 'noise', 'saturation'],
        'num_corrupts_per_image': 1
    }

    ds = SingleImageDataset(opt)
    import os
    os.makedirs("debug", exist_ok=True)
    for i in range(0, len(ds)):
        o = ds[i]
        for k, v in o.items():
            if 'path' not in k and 'center' not in k:
                #if 'full' in k:
                    #masked = v[:3, :, :] * v[3]
                    #torchvision.utils.save_image(masked.unsqueeze(0), "debug/%i_%s_masked.png" % (i, k))
                    #v = v[:3, :, :]
                import torchvision
                torchvision.utils.save_image(v.unsqueeze(0), "debug/%i_%s.png" % (i, k))