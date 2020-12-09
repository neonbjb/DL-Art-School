import torch
from torch.utils.data import Dataset


# Dataset that feeds random data into the state. Can be useful for testing or demo purposes without actual data.
class RandomDataset(Dataset):
    def __init__(self, opt):
        self.hq_shape = tuple(opt['hq_shape'])
        self.lq_shape = tuple(opt['lq_shape'])

    def __getitem__(self, item):
        return {'lq': torch.rand(self.lq_shape), 'hq': torch.rand(self.hq_shape),
                'LQ_path': '', 'GT_path': ''}

    def __len__(self):
        # Arbitrary
        return 1024 * 1024
