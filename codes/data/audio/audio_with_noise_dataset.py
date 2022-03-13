import random
import sys
from math import pi

import librosa
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

import torch.nn.functional as F
from data.audio.unsupervised_audio_dataset import UnsupervisedAudioDataset, load_audio
from data.util import load_paths_from_cache, find_files_of_type, is_audio_file

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


def load_rir(path, sr, max_sz):
    rir = load_audio(path, sr).abs()
    if rir.shape[-1] > max_sz:
        rir = rir[:, :max_sz]
    rir = (rir / torch.norm(rir, p=2)).flip([1])
    return rir


'''
Wraps a unsupervised_audio_dataset and applies noise to the output clips, then provides labels depending on what
noise was added.
'''
class AudioWithNoiseDataset(Dataset):
    def __init__(self, opt):
        self.underlying_dataset = UnsupervisedAudioDataset(opt)
        self.env_noise_paths = load_paths_from_cache(opt['env_noise_paths'], opt['env_noise_cache'])
        self.music_paths = load_paths_from_cache(opt['music_paths'], opt['music_cache'])
        self.openair_paths = find_files_of_type('img', opt['openair_path'], qualifier=is_audio_file)[0]
        self.min_volume = opt_get(opt, ['min_noise_volume'], .2)
        self.max_volume = opt_get(opt, ['max_noise_volume'], .5)
        self.sampling_rate = self.underlying_dataset.sampling_rate
        self.use_gpu_for_reverb_compute = opt_get(opt, ['use_gpu_for_reverb_compute'], True)
        self.openair_kernels = None
        self.current_item_fetch = 0
        self.fetch_error_count = 0

    def load_openair_kernels(self):
        if self.use_gpu_for_reverb_compute and self.openair_kernels is None:
            # Load the openair reverbs as CUDA tensors.
            self.openair_kernels = []
            for oa in self.openair_paths:
                self.openair_kernels.append(load_rir(oa, self.underlying_dataset.sampling_rate, self.underlying_dataset.sampling_rate*2).cuda())

    def __getitem__(self, item):
        if self.current_item_fetch != item:
            self.current_item_fetch = item
            self.fetch_error_count = 0

        # Load on the fly to prevent GPU memory sharing across process errors.
        self.load_openair_kernels()

        out = self.underlying_dataset[item]
        clip = out['clip']
        dlen = clip.shape[-1]
        clip = clip[:, :out['clip_lengths']]
        padding_room = dlen - clip.shape[-1]
        augpath = ''
        augvol = 0
        try:
            # Randomly adjust clip volume, regardless of the selection, between
            clipvol = (random.random() * (.8-.5) + .5)
            clip = clip * clipvol

            label = random.randint(0, 4)  # Current excludes GSM corruption.
            #label = 3
            if label > 0 and label < 4:  # 0 is basically "leave it alone"
                aug_needed = True
                augvol = (random.random() * (self.max_volume-self.min_volume) + self.min_volume)
                if label == 1:
                    # Add environmental noise.
                    augpath = random.choice(self.env_noise_paths)
                    intg_fns = [_integration_fn_fully_enabled]
                elif label == 2:
                    # Add music
                    augpath = random.choice(self.music_paths)
                    intg_fns = [_integration_fn_fully_enabled]
                    augvol *= .5  # Music is often severely in the background.
                elif label == 3:
                    augpath = random.choice(self.underlying_dataset.audiopaths)
                    # This can take two forms:
                    if padding_room < 22000 or random.random() < .5:
                        # (1) The voices talk over one another. If there is no padding room, we always take this choice.
                        intg_fns = [_integration_fn_smooth, _integration_fn_fully_enabled]
                    else:
                        # (2) There are simply two voices in the clip, separated from one another.
                        # This is a special case that does not use the same logic as the rest of the augmentations.
                        aug = load_audio(augpath, self.underlying_dataset.sampling_rate)
                        # Pad with some random silence
                        aug = F.pad(aug, (random.randint(20,4000), 0))
                        # Fit what we can given the padding room we have.
                        aug = aug[:, :padding_room]
                        clip = torch.cat([clip, aug], dim=1)
                        # Restore some meta-parameters.
                        padding_room = dlen - clip.shape[-1]
                        out['clip_lengths'] = torch.tensor(clip.shape[-1])
                        aug_needed = False
                if aug_needed:
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
            elif label == 4:
                # Perform reverb (to simulate being in a large room with an omni-mic). This is performed by convolving
                # impulse recordings from openair over the input clip.
                if self.use_gpu_for_reverb_compute:
                    rir = random.choice(self.openair_kernels)
                else:
                    augpath = random.choice(self.openair_paths)
                    rir = load_rir(augpath, self.underlying_dataset.sampling_rate, clip.shape[-1])
                clip = torch.nn.functional.pad(clip, (rir.shape[1]-1, 0))
                if self.use_gpu_for_reverb_compute:
                    clip = clip.cuda()
                clip = torch.nn.functional.conv1d(clip.unsqueeze(0), rir.unsqueeze(0)).squeeze(0).cpu()
            elif label == 5:
                # Apply the GSM codec to simulate cellular phone audio.
                clip = torchaudio.functional.apply_codec(clip, self.underlying_dataset.sampling_rate, format="gsm")
        except:
            if self.fetch_error_count > 10:
                print(f"Exception encountered processing {item}, re-trying because this is often just a failed aug.")
                print(sys.exc_info())
                #raise  # Uncomment to surface exceptions.
            self.fetch_error_count += 1
            return self[item]

        clip.clip_(-1, 1)
        # Restore padding.
        clip = F.pad(clip, (0, padding_room))
        out['clip'] = clip
        out['label'] = label
        #out['aug'] = aug
        out['augpath'] = augpath
        out['augvol'] = augvol
        out['clipvol'] = clipvol
        return out

    def __len__(self):
        return len(self.underlying_dataset)


if __name__ == '__main__':
    params = {
        'mode': 'unsupervised_audio_with_noise',
        'path': ['y:/clips/books1'],
        'cache_path': 'D:\\data\\clips_for_noise_classifier.pth',
        'sampling_rate': 22050,
        'pad_to_samples': 400000,
        'do_augmentation': False,
        'phase': 'train',
        'n_workers': 4,
        'batch_size': 256,
        'extra_samples': 0,
        'env_noise_paths': ['E:\\audio\\UrbanSound\\filtered', 'E:\\audio\\UrbanSound\\MSSND'],
        'env_noise_cache': 'E:\\audio\\UrbanSound\\cache.pth',
        'music_paths': ['E:\\audio\\music\\FMA\\fma_large', 'E:\\audio\\music\\maestro\\maestro-v3.0.0'],
        'music_cache': 'E:\\audio\\music\\cache.pth',
        'openair_path': 'D:\\data\\audio\\openair\\resampled',
        'use_gpu_for_reverb_compute': False,
    }
    from data import create_dataset, create_dataloader, util

    ds = create_dataset(params)
    dl = create_dataloader(ds, params, pin_memory=False)
    i = 0
    for b in tqdm(dl):
        for b_ in range(b['clip'].shape[0]):
            #torchaudio.save(f'{i}_clip_{b_}_{b["label"][b_].item()}.wav', b['clip'][b_][:, :b['clip_lengths'][b_]], ds.sampling_rate)
            #torchaudio.save(f'{i}_clip_{b_}_aug.wav', b['aug'][b_], ds.sampling_rate)
            print(f'{i} aug path: {b["augpath"][b_]} aug volume: {b["augvol"][b_]} clip volume: {b["clipvol"][b_]}')
            i += 1
