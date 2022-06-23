import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
import torchvision
from tqdm import tqdm

from utils.util import opt_get


class PreprocessedMelDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        path = opt['path']
        cache_path = opt['cache_path']  # Will fail when multiple paths specified, must be specified in this case.
        if os.path.exists(cache_path):
            self.paths = torch.load(cache_path)
        else:
            print("Building cache..")
            path = Path(path)
            self.paths = [str(p) for p in path.rglob("*.npz")]
            torch.save(self.paths, cache_path)
        self.pad_to = opt_get(opt, ['pad_to_samples'], 10336)
        self.squeeze = opt_get(opt, ['should_squeeze'], False)

    def __getitem__(self, index):
        with np.load(self.paths[index]) as npz_file:
            mel = torch.tensor(npz_file['arr_0'])
        assert mel.shape[-1] <= self.pad_to
        if self.squeeze:
            mel = mel.squeeze()
        padding_needed = self.pad_to - mel.shape[-1]
        mask = torch.zeros_like(mel)
        if padding_needed > 0:
            mel = F.pad(mel, (0,padding_needed))
            mask = F.pad(mask, (0,padding_needed), value=1)

        output = {
            'mel': mel,
            'mel_lengths': torch.tensor(mel.shape[-1]),
            'mask': mask,
            'mask_lengths': torch.tensor(mask.shape[-1]),
            'path': self.paths[index],
        }
        return output

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    params = {
        'mode': 'preprocessed_mel',
        'path': 'Y:\\separated\\large_mel_cheaters',
        'cache_path': 'Y:\\separated\\large_mel_cheaters_win.pth',
        'pad_to_samples': 646,
        'phase': 'train',
        'n_workers': 0,
        'batch_size': 16,
    }
    from data import create_dataset, create_dataloader

    ds = create_dataset(params)
    dl = create_dataloader(ds, params)
    i = 0
    for b in tqdm(dl):
        #pass
        torchvision.utils.save_image((b['mel'].unsqueeze(1)+1)/2, f'{i}.png')
        i += 1
        if i > 20:
            break
