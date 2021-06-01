import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision import datasets

# Wrapper for basic pytorch datasets which re-wraps them into a format usable by ExtensibleTrainer.
from utils.util import opt_get


class TorchDataset(Dataset):
    def __init__(self, opt):
        DATASET_MAP = {
            "mnist": datasets.MNIST,
            "fmnist": datasets.FashionMNIST,
            "cifar10": datasets.CIFAR10,
            "imagenet": datasets.ImageNet,
            "imagefolder": datasets.ImageFolder
        }
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if opt_get(opt, ['random_crop'], False):
            transforms = [
                T.RandomResizedCrop(opt['image_size']),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ]
        else:
            transforms = [
                T.Resize(opt['image_size']),
                T.CenterCrop(opt['image_size']),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ]
        transforms = T.Compose(transforms)
        self.dataset = DATASET_MAP[opt['dataset']](transform=transforms, **opt['kwargs'])
        self.len = opt['fixed_len'] if 'fixed_len' in opt.keys() else len(self.dataset)

    def __getitem__(self, item):
        underlying_item, lbl = self.dataset[item]
        return {'lq': underlying_item, 'hq': underlying_item, 'labels': lbl,
                'LQ_path': str(item), 'GT_path': str(item)}

    def __len__(self):
        return self.len

if __name__ == '__main__':
    opt = {
        'flip': True,
        'crop_sz': None,
        'dataset': 'imagefolder',
        'resize': 256,
        'center_crop': 224,
        'normalize': True,
        'kwargs': {
            'root': 'F:\\4k6k\\datasets\\images\\imagenet_2017\\val',
        }
    }
    set = TorchDataset(opt)
    j = set[0]
