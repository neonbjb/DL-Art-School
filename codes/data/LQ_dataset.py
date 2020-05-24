import numpy as np
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import torchvision.transforms.functional as F
from PIL import Image
import os.path as osp


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
        self.paths_LQ, self.paths_GT = None, None
        self.LQ_env = None  # environment for lmdb

        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        self.paths_LQ = self.paths_LQ[self.start_at:]
        assert self.paths_LQ, 'Error: LQ paths are empty.'

    def _init_lmdb(self):
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and self.LQ_env is None:
            self._init_lmdb()
        actual_index = index # int(index / 2)
        is_left = (index % 2) == 0

        # get LQ image
        LQ_path = self.paths_LQ[actual_index]
        img_LQ = Image.open(LQ_path)
        left = 0 if is_left else 1920
        # crop input if needed.
        #img_LQ = F.crop(img_LQ, 5, left + 5, 1900, 1900)
        img_LQ = F.to_tensor(img_LQ)

        img_name = osp.splitext(osp.basename(LQ_path))[0]
        LQ_path = LQ_path.replace(img_name, img_name + "_%i" % (index % 2))

        return {'LQ': img_LQ, 'LQ_path': LQ_path}

    def __len__(self):
        return len(self.paths_LQ) # * 2
