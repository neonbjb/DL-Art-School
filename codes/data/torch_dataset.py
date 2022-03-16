from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision import datasets

# Wrapper for basic pytorch datasets which re-wraps them into a format usable by ExtensibleTrainer.
from data.images.cifar import CIFAR100, CIFAR10
from utils.util import opt_get


class TorchDataset(Dataset):
    def __init__(self, opt):
        DATASET_MAP = {
            "mnist": datasets.MNIST,
            "fmnist": datasets.FashionMNIST,
            "cifar10": CIFAR10,
            "cifar100": CIFAR100,
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
        self.len = opt_get(opt, ['fixed_len'], len(self.dataset))
        self.offset = opt_get(opt, ['offset'], 0)

    def __getitem__(self, item):
        item = self.dataset[item+self.offset]
        if len(item) == 2:
            underlying_item, lbl = item
            coarselbl = None
        elif len(item) == 3:
            underlying_item, lbl, coarselbl = item
        else:
            raise NotImplementedError
        return {'lq': underlying_item, 'hq': underlying_item, 'labels': lbl, 'coarse_labels': coarselbl,
                'LQ_path': str(item), 'GT_path': str(item)}

    def __len__(self):
        return self.len-self.offset

if __name__ == '__main__':
    opt = {
        'flip': True,
        'crop_sz': None,
        'dataset': 'cifar100',
        'image_size': 32,
        'normalize': True,
        'kwargs': {
            'root': 'E:\\4k6k\\datasets\\images\\cifar100',
            'download': True
        }
    }
    set = TorchDataset(opt)
    j = set[0]
