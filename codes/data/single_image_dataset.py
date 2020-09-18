from torch.utils import data
from data.chunk_with_reference import ChunkWithReference
from data.image_corruptor import ImageCorruptor
import os
from bisect import bisect_left
import cv2
import torch


# Builds a dataset composed of a set of folders. Each folder represents a single high resolution image that has been
# chunked into patches of fixed size. A reference image is included as well as a list of center points for each patch.
class SingleImageDataset(data.Dataset):

    def __init__(self, opt):
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

        # Indexing this dataset is tricky. Aid it by having a sorted list of starting indices for each chunk.
        start = 0
        self.starting_indices = []
        for c in chunks:
            self.starting_indices.append(start)
            start += len(c)
        self.len = start

    def binary_search(elem, sorted_list):
        # https://docs.python.org/3/library/bisect.html
        'Locate the leftmost value exactly equal to x'
        i = bisect_left(sorted_list, elem)
        if i != len(sorted_list) and sorted_list[i] == elem:
            return i
        return -1

    def resize_point(self, point, orig_dim, new_dim):
        oh, ow = orig_dim
        nh, nw = new_dim
        dh, dw = float(nh) / float(oh), float(nw) / float(ow)
        point[0] = int(dh * float(point[0]))
        point[1] = int(dw * float(point[1]))
        return point

    def __getitem__(self, item):
        chunk_ind = self.binary_search(item, self.starting_indices)
        hq, hq_ref, hq_center, path = self.chunks[item-self.starting_indices[chunk_ind]]

        # Enforce size constraints
        h, w, _ = hq.shape
        if self.target_hq_size is not None and self.target_hq_size != h:
            # It is assumed that the target size is a square.
            target_size = (self.target_hq_size, self.target_hq_size)
            hq = cv2.resize(hq, target_size, interpolation=cv2.INTER_LINEAR)
            hq_ref = cv2.resize(hq_ref, target_size, interpolation=cv2.INTER_LINEAR)
            hq_center = self.resize_point(hq_center, (h, w), target_size)
            h, w = self.target_hq_size, self.target_hq_size
        hq_multiple = self.multiple * self.scale   # Multiple must apply to LQ image.
        if h % hq_multiple != 0 or w % hq_multiple != 0:
            h, w = (h - h % hq_multiple), (w - w % hq_multiple)
            hq_center = self.resize_point(hq_center, hq.shape[:1], (h, w))
            hq = hq[:h, :w, :]
            hq_ref = hq_ref[:h, :w, :]

        # Synthesize the LQ image
        if self.for_eval:
            lq, lq_ref = hq, hq_ref
        else:
            lq = cv2.resize(hq, (h // self.scale, w // self.scale), interpolation=cv2.INTER_LINEAR)
            lq_ref = cv2.resize(hq_ref, (h // self.scale, w // self.scale), interpolation=cv2.INTER_LINEAR)
            lq_center = self.resize_point(hq_center, (h, w), lq.shape[:1])

        # Corrupt the LQ image
        lq = self.corruptor.corrupt_images([lq])

        # Convert to torch tensor
        hq = torch.from_numpy(np.ascontiguousarray(np.transpose(hq, (2, 0, 1)))).float()
        hq_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(hq_ref, (2, 0, 1)))).float()
        lq = F.to_tensor(lq)
        lq_ref = F.to_tensor(lq_ref)

        return {'LQ': lq, 'GT': hq, 'gt_fullsize_ref': hq_ref, 'lq_fullsize_ref': lq_ref,
             'lq_center': lq_center, 'gt_center': hq_center,
             'LQ_path': path, 'GT_path': path}

    def __len__(self):
        return self.len