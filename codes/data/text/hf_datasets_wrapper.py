from torch.utils.data import Dataset
import datasets


class HfDataset(Dataset):
    """
    Simple wrapper for a HuggingFace dataset that can re-map keys if desired.
    """
    def __init__(self, corpi, cache_path=None, key_maps=None, dataset_spec_key='train'):
        self.hfd = []
        for corpus in corpi:
            dataset_name, config = corpus
            if config == '' or config == 'None':
                config = None
            self.hfd.append(datasets.load_dataset(dataset_name, config, cache_dir=cache_path)[dataset_spec_key])
        self.key_maps = key_maps

    def __getitem__(self, item):
        for dataset in self.hfd:
            if item < len(dataset):
                val = dataset[item]
                if self.key_maps is None:
                    return val
                else:
                    return {k: val[v] for k, v in self.key_maps.items()}
            else:
                item -= len(dataset)
        raise IndexError()

    def __len__(self):
        return sum([len(h) for h in self.hfd])


if __name__ == '__main__':
    d = HfDataset([['wikipedia', '20200501.en'], ['bookcorpus', '']], dataset_spec_key='train', cache_path='Z:\\huggingface_datasets\\cache')
    print(d[5])
