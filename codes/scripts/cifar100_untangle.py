import numpy
import torch
from torch.utils.data import DataLoader

from data.torch_dataset import TorchDataset
from models.classifiers.cifar_resnet_branched import ResNet
from models.classifiers.cifar_resnet_branched import BasicBlock

if __name__ == '__main__':
    dopt = {
        'flip': True,
        'crop_sz': None,
        'dataset': 'cifar100',
        'image_size': 32,
        'normalize': False,
        'kwargs': {
            'root': 'E:\\4k6k\\datasets\\images\\cifar100',
            'download': True
        }
    }
    set = TorchDataset(dopt)
    loader = DataLoader(set, num_workers=0, batch_size=32)
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load('C:\\Users\\jbetk\\Downloads\\cifar_hardw_10000.pth'))
    model.eval()

    bins = [[] for _ in range(8)]
    for i, batch in enumerate(loader):
        logits, selector = model(batch['hq'], coarse_label=None, return_selector=True)
        for k, s in enumerate(selector):
            for j, b in enumerate(s):
                if b:
                    bins[j].append(batch['labels'][k].item())
        if i > 10:
            break

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3,3)
    for i in range(8):
        axs[i%3, i//3].hist(numpy.asarray(bins[i]))
    plt.show()
    print('hi')