from data.images.base_unsupervised_image_dataset import BaseUnsupervisedImageDataset
import numpy as np
import torch
from bisect import bisect_left
import os.path as osp

class MultiFrameDataset(BaseUnsupervisedImageDataset):
    def __init__(self, opt):
        super(MultiFrameDataset, self).__init__(opt)
        self.num_frames = opt['num_frames']

    def chunk_name(self, i):
        return osp.basename(self.chunks[i].path)

    def get_sequential_image_paths_from(self, chunk_index, chunk_offset):
        im_name = self.chunk_name(chunk_index)
        source_name = im_name[:-12]
        # Search backwards for the frames needed. We are assuming that every video in the dataset has at least frames_needed frames.
        frames_needed = self.num_frames
        search_idx = chunk_index
        while frames_needed > 0:
            frames_needed -= 1
            search_idx -= 1
            if source_name not in self.chunk_name(search_idx):
                search_idx += 1
                break

        # Now build num_frames starting from search_idx.
        hqs, refs, masks, centers = [], [], [], []
        for i in range(self.num_frames):
            idx = search_idx + i
            if idx < 0 or idx >= len(self.chunks) or chunk_offset < 0 or chunk_offset >= len(self.chunks[idx]):
                print("Chunk reference indexing failed for %s." % (im_name,), search_idx, i, chunk_offset, self.num_frames)
            h, r, c, m, p = self.chunks[search_idx + i][chunk_offset]
            hqs.append(h)
            refs.append(r)
            masks.append(m)
            centers.append(c)
            path = p
        return hqs, refs, masks, centers, path

    def __getitem__(self, item):
        chunk_ind = bisect_left(self.starting_indices, item)
        chunk_ind = chunk_ind if chunk_ind < len(self.starting_indices) and self.starting_indices[chunk_ind] == item else chunk_ind-1
        hqs, refs, masks, centers, path = self.get_sequential_image_paths_from(chunk_ind, item-self.starting_indices[chunk_ind])

        hs, hrs, hms, hcs = self.resize_hq(hqs, refs, masks, centers)
        ls, lrs, lms, lcs = self.synthesize_lq(hs, hrs, hms, hcs)

        # Convert to torch tensor
        hq = torch.from_numpy(np.ascontiguousarray(np.transpose(np.stack(hs), (0, 3, 1, 2)))).float()
        hq_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(np.stack(hrs), (0, 3, 1, 2)))).float()
        hq_mask = torch.from_numpy(np.ascontiguousarray(np.stack(hms))).unsqueeze(dim=1)
        hq_ref = torch.cat([hq_ref, hq_mask], dim=1)
        lq = torch.from_numpy(np.ascontiguousarray(np.transpose(np.stack(ls), (0, 3, 1, 2)))).float()
        lq_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(np.stack(lrs), (0, 3, 1, 2)))).float()
        lq_mask = torch.from_numpy(np.ascontiguousarray(np.stack(lms))).unsqueeze(dim=1)
        lq_ref = torch.cat([lq_ref, lq_mask], dim=1)

        return {'GT_path': path, 'lq': lq, 'hq': hq, 'gt_fullsize_ref': hq_ref, 'lq_fullsize_ref': lq_ref,
             'lq_center': torch.tensor(lcs, dtype=torch.long), 'gt_center': torch.tensor(hcs, dtype=torch.long)}


if __name__ == '__main__':
    opt = {
        'name': 'amalgam',
        'paths': ['F:\\4k6k\\datasets\\images\\fullvideo\\ge_fv_256_tiled'],
        'weights': [1],
        'target_size': 128,
        'force_multiple': 32,
        'scale': 2,
        'eval': False,
        'fixed_corruptions': ['jpeg-medium'],
        'random_corruptions': [],
        'num_corrupts_per_image': 0,
        'num_frames': 10
    }

    ds = MultiFrameDataset(opt)
    import os
    os.makedirs("debug", exist_ok=True)
    bs = 0
    batch = None
    for i in range(len(ds)):
        import random
        k = 'lq'
        element = ds[random.randint(0,len(ds))]
        base_file = osp.basename(element["GT_path"])
        o = element[k].unsqueeze(0)
        if bs < 32:
            if batch is None:
                batch = o
            else:
                batch = torch.cat([batch, o], dim=0)
            bs += 1
            continue

        if 'path' not in k and 'center' not in k:
            b, fr, f, h, w = batch.shape
            for j in range(fr):
                import torchvision
                base=osp.basename(base_file)
                torchvision.utils.save_image(batch[:, j], "debug/%i_%s_%i__%s.png" % (i, k, j, base))

        bs = 0
        batch = None
