import torch
from data import create_dataset


# Simple composite dataset that combines multiple other datasets.
# Assumes that the datasets output dicts.
class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.datasets = {}
        for k, v in opt.items():
            if not isinstance(v, dict):
                continue
            # Scale&phase gets injected by options.py..
            v['scale'] = opt['scale']
            v['phase'] = opt['phase']
            self.datasets[k] = create_dataset(v)
        self.items_fetched = 0

    def __getitem__(self, i):
        self.items_fetched += 1
        output = {}
        for name, dataset in self.datasets.items():
            prefix = ""
            # 'default' dataset gets no prefix, other ones get `key_`
            if name != 'default':
                prefix = name + "_"

            data = dataset[i % len(dataset)]
            for k, v in data.items():
                output[prefix + k] = v
        return output

    def __len__(self):
        return max(len(d) for d in self.datasets.values())