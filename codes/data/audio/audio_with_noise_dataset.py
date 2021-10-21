import random
from math import pi

import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from data.audio.unsupervised_audio_dataset import UnsupervisedAudioDataset, load_audio
from data.util import load_paths_from_cache


# Just all ones.
from utils.util import opt_get


def _integration_fn_fully_enabled(n):
    return torch.ones((n,))


# Randomly assigns up to 5 blocks of the output tensor the value '1'. Rest is zero
def _integration_fn_spiky(n):
    fn = torch.zeros((n,))
    spikes = random.randint(1,5)
    for _ in range(spikes):
        sz = random.randint(n//8, n//2)
        pos = random.randint(0, n)
        extent = min(n, sz+pos)
        fn[pos:extent] = 1
    return fn


# Uses a sinusoidal ramp up and down (of random length) to a peak which is held for a random duration.
def _integration_fn_smooth(n):
    center = random.randint(1, n-2)
    max_duration=n-center-1
    duration = random.randint(max_duration//4, max_duration)
    end = center+duration

    ramp_up_sz = random.randint(n//16,n//4)
    ramp_up = torch.sin(pi*torch.arange(0,ramp_up_sz)/(2*ramp_up_sz))
    if ramp_up_sz > center:
        ramp_up = ramp_up[(ramp_up_sz-center):]
        ramp_up_sz = center

    ramp_down_sz = random.randint(n//16,n//4)
    ramp_down = torch.flip(torch.sin(pi*torch.arange(0,ramp_down_sz)/(2*ramp_down_sz)), dims=[0])
    if ramp_down_sz > (n-end):
        ramp_down = ramp_down[:(n-end)]
        ramp_down_sz = n-end

    fn = torch.zeros((n,))
    fn[(center-ramp_up_sz):center] = ramp_up
    fn[center:end] = 1
    fn[end:(end+ramp_down_sz)] = ramp_down

    return fn


'''
Wraps a unsupervised_audio_dataset and applies noise to the output clips, then provides labels depending on what
noise was added.
'''
class AudioWithNoiseDataset(Dataset):
    def __init__(self, opt):
        self.underlying_dataset = UnsupervisedAudioDataset(opt)
        self.env_noise_paths = load_paths_from_cache(opt['env_noise_paths'], opt['env_noise_cache'])
        self.music_paths = load_paths_from_cache(opt['music_paths'], opt['music_cache'])
        self.min_volume = opt_get(opt, ['min_noise_volume'], .2)
        self.max_volume = opt_get(opt, ['max_noise_volume'], .5)
        self.sampling_rate = self.underlying_dataset.sampling_rate

    def __getitem__(self, item):
        out = self.underlying_dataset[item]
        clip = out['clip']
        augpath = ''
        augvol = 0
        try:
            # Randomly adjust clip volume, regardless of the selection, between
            clipvol = (random.random() * (.8-.5) + .5)
            clip = clip * clipvol

            label = random.randint(0,3)
            aug = torch.zeros_like(clip)
            if label != 0:  # 0 is basically "leave it alone"
                augvol = (random.random() * (self.max_volume-self.min_volume) + self.min_volume)
                if label == 1:
                    augpath = random.choice(self.env_noise_paths)
                    intg_fns = [_integration_fn_fully_enabled]
                elif label == 2:
                    augpath = random.choice(self.music_paths)
                    intg_fns = [_integration_fn_fully_enabled]
                    augvol *= .5  # Music is often severely in the background.
                elif label == 3:
                    augpath = random.choice(self.underlying_dataset.audiopaths)
                    intg_fns = [_integration_fn_smooth, _integration_fn_fully_enabled]
                aug = load_audio(augpath, self.underlying_dataset.sampling_rate)
                if aug.shape[1] > clip.shape[1]:
                    n, cn = aug.shape[1], clip.shape[1]
                    gap = n-cn
                    placement = random.randint(0, gap)
                    aug = aug[:, placement:placement+cn]
                aug = random.choice(intg_fns)(aug.shape[1]) * aug
                aug = aug * augvol
                if aug.shape[1] < clip.shape[1]:
                    gap = clip.shape[1] - aug.shape[1]
                    placement = random.randint(0, gap-1)
                    aug = torch.nn.functional.pad(aug, (placement, gap-placement))
                clip = clip + aug
                clip.clip_(-1, 1)
        except:
            print("Exception encountered processing {item}, re-trying because this is often just a failed aug.")
            return self[item]

        out['clip'] = clip
        out['label'] = label
        out['aug'] = aug
        out['augpath'] = augpath
        out['augvol'] = augvol
        out['clipvol'] = clipvol
        return out

    def __len__(self):
        return len(self.underlying_dataset)


if __name__ == '__main__':
    params = {
        'mode': 'unsupervised_audio_with_noise',
        'path': ['\\\\192.168.5.3\\rtx3080_audio_y\\split\\books2', '\\\\192.168.5.3\\rtx3080_audio\\split\\books1', '\\\\192.168.5.3\\rtx3080_audio\\split\\cleaned-2'],
        'cache_path': 'E:\\audio\\remote-cache2.pth',
        'sampling_rate': 22050,
        'pad_to_samples': 80960,
        'phase': 'train',
        'n_workers': 0,
        'batch_size': 16,
        'extra_samples': 4,
        'env_noise_paths': ['E:\\audio\\UrbanSound\\filtered', 'E:\\audio\\UrbanSound\\MSSND'],
        'env_noise_cache': 'E:\\audio\\UrbanSound\\cache.pth',
        'music_paths': ['E:\\audio\\music\\FMA\\fma_large', 'E:\\audio\\music\\maestro\\maestro-v3.0.0'],
        'music_cache': 'E:\\audio\\music\\cache.pth',
    }
    from data import create_dataset, create_dataloader, util

    ds = create_dataset(params)
    dl = create_dataloader(ds, params)
    i = 0
    for b in tqdm(dl):
        for b_ in range(b['clip'].shape[0]):
            #pass
            torchaudio.save(f'{i}_clip_{b_}_{b["label"][b_].item()}.wav', b['clip'][b_], ds.sampling_rate)
            torchaudio.save(f'{i}_clip_{b_}_aug.wav', b['aug'][b_], ds.sampling_rate)
            print(f'{i} aug path: {b["augpath"][b_]} aug volume: {b["augvol"][b_]} clip volume: {b["clipvol"][b_]}')
            i += 1
