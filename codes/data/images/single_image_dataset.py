import random
from bisect import bisect_left
import numpy as np
import torch
from data.images.base_unsupervised_image_dataset import BaseUnsupervisedImageDataset


# Builds a dataset composed of a set of folders. Each folder represents a single high resolution image that has been
# chunked into patches of fixed size. A reference image is included as well as a list of center points for each patch.
class SingleImageDataset(BaseUnsupervisedImageDataset):
    def __init__(self, opt):
        super(SingleImageDataset, self).__init__(opt)

    def get_paths(self):
        for i in range(len(self)):
            chunk_ind = bisect_left(self.starting_indices, i)
            chunk_ind = chunk_ind if chunk_ind < len(self.starting_indices) and self.starting_indices[chunk_ind] == i else chunk_ind-1
            yield self.chunks[chunk_ind].tiles[i-self.starting_indices[chunk_ind]]

    def __getitem__(self, item):
        chunk_ind = bisect_left(self.starting_indices, item)
        chunk_ind = chunk_ind if chunk_ind < len(self.starting_indices) and self.starting_indices[chunk_ind] == item else chunk_ind-1
        hq, hq_ref, hq_center, hq_mask, path = self.chunks[chunk_ind][item-self.starting_indices[chunk_ind]]

        hs, hrs, hms, hcs = self.resize_hq([hq], [hq_ref], [hq_mask], [hq_center])
        ls, lrs, lms, lcs = self.synthesize_lq(hs, hrs, hms, hcs)

        # Convert to torch tensor
        hq = torch.from_numpy(np.ascontiguousarray(np.transpose(hs[0], (2, 0, 1)))).float()
        hq_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(hrs[0], (2, 0, 1)))).float()
        hq_mask = torch.from_numpy(np.ascontiguousarray(hms[0])).unsqueeze(dim=0)
        hq_ref = torch.cat([hq_ref, hq_mask], dim=0)
        lq = torch.from_numpy(np.ascontiguousarray(np.transpose(ls[0], (2, 0, 1)))).float()
        lq_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(lrs[0], (2, 0, 1)))).float()
        lq_mask = torch.from_numpy(np.ascontiguousarray(lms[0])).unsqueeze(dim=0)
        lq_ref = torch.cat([lq_ref, lq_mask], dim=0)

        return {'lq': lq, 'hq': hq, 'gt_fullsize_ref': hq_ref, 'lq_fullsize_ref': lq_ref,
             'lq_center': torch.tensor(lcs[0], dtype=torch.long), 'gt_center': torch.tensor(hcs[0], dtype=torch.long),
             'LQ_path': path, 'GT_path': path}


if __name__ == '__main__':
    opt = {
        'name': 'amalgam',
        'paths': ['F:\\4k6k\\datasets\\images\\mi1_256'],
        'weights': [1],
        'target_size': 128,
        'force_multiple': 32,
        'scale': 2,
        'eval': False,
        'fixed_corruptions': ['jpeg-broad', 'gaussian_blur'],
        'random_corruptions': ['noise-5', 'none'],
        'num_corrupts_per_image': 1,
        'corrupt_before_downsize': True,
    }

    ds = SingleImageDataset(opt)
    import os
    os.makedirs("debug", exist_ok=True)
    for i in range(0, len(ds)):
        o = ds[random.randint(0, len(ds))]
        #for k, v in o.items():
        k = 'lq'
        v = o[k]
        #if 'LQ' in k and 'path' not in k and 'center' not in k:
        #if 'full' in k:
            #masked = v[:3, :, :] * v[3]
            #torchvision.utils.save_image(masked.unsqueeze(0), "debug/%i_%s_masked.png" % (i, k))
            #v = v[:3, :, :]
        import torchvision
        torchvision.utils.save_image(v.unsqueeze(0), "debug/%i_%s.png" % (i, k))