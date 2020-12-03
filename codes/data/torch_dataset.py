import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision import datasets

# Wrapper for basic pytorch datasets which re-wraps them into a format usable by ExtensibleTrainer.
class TorchDataset(Dataset):
    def __init__(self, opt):
        DATASET_MAP = {
            "mnist": datasets.MNIST,
            "fmnist": datasets.FashionMNIST,
            "cifar10": datasets.CIFAR10,
        }
        transforms = []
        if opt['flip']:
            transforms.append(T.RandomHorizontalFlip())
        if opt['crop_sz']:
            transforms.append(T.RandomCrop(opt['crop_sz'], padding=opt['padding'], padding_mode="reflect"))
        transforms.append(T.ToTensor())
        transforms = T.Compose(transforms)
        is_for_training = opt['test'] if 'test' in opt.keys() else True
        self.dataset = DATASET_MAP[opt['dataset']](opt['datapath'], train=is_for_training, download=True, transform=transforms)
        self.len = opt['fixed_len'] if 'fixed_len' in opt.keys() else len(self.dataset)

    def __getitem__(self, item):
        underlying_item = self.dataset[item][0]
        return {'LQ': underlying_item, 'GT': underlying_item,
                'LQ_path': str(item), 'GT_path': str(item)}

    def __len__(self):
        return self.len
