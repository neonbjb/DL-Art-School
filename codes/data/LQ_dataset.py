import numpy as np
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import torchvision.transforms.functional as F
from PIL import Image
import os.path as osp
import cv2


class LQDataset(data.Dataset):
    '''Read LQ images only in the test phase.'''

    def __init__(self, opt):
        super(LQDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        if 'start_at' in self.opt.keys():
            self.start_at = self.opt['start_at']
        else:
            self.start_at = 0
        self.vertical_splits = self.opt['vertical_splits']
        self.paths_LQ, self.paths_GT = None, None
        self.LQ_env = None  # environment for lmdb
        self.force_multiple = self.opt['force_multiple'] if 'force_multiple' in self.opt.keys() else 1

        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        self.paths_LQ = self.paths_LQ[self.start_at:]
        assert self.paths_LQ, 'Error: LQ paths are empty.'

    def _init_lmdb(self):
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and self.LQ_env is None:
            self._init_lmdb()
        if self.vertical_splits > 0:
            actual_index = int(index / self.vertical_splits)
        else:
            actual_index = index

        # get LQ image
        LQ_path = self.paths_LQ[actual_index]
        img_LQ = Image.open(LQ_path)
        if self.vertical_splits > 0:
            w, h = img_LQ.size
            split_index = (index % self.vertical_splits)
            w_per_split = int(w / self.vertical_splits)
            left = w_per_split * split_index
            img_LQ = F.crop(img_LQ, 0, left, h, w_per_split)

        # Enforce force_resize constraints.
        h, w = img_LQ.size
        if h % self.force_multiple != 0 or w % self.force_multiple != 0:
            h, w = (w - w % self.force_multiple), (h - h % self.force_multiple)
            img_LQ = img_LQ.resize((w, h))

        img_LQ = F.to_tensor(img_LQ)

        img_name = osp.splitext(osp.basename(LQ_path))[0]
        LQ_path = LQ_path.replace(img_name, img_name + "_%i" % (index % self.vertical_splits))

        return {'LQ': img_LQ, 'LQ_path': LQ_path}

    def __len__(self):
        if self.vertical_splits > 0:
            return len(self.paths_LQ) * self.vertical_splits
        else:
            return len(self.paths_LQ)
