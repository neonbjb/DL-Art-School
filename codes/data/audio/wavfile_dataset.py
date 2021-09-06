import os
import random

import torch
import torch.utils.data
import torchaudio
from tqdm import tqdm

from data.audio.wav_aug import WavAugmentor
from data.util import find_files_of_type, is_wav_file
from models.tacotron2.taco_utils import load_wav_to_torch
from utils.util import opt_get


def load_audio_from_wav(audiopath, sampling_rate):
        audio, lsr = load_wav_to_torch(audiopath)
        if lsr != sampling_rate:
            if lsr < sampling_rate:
                print(f'{audiopath} has a sample rate of {sampling_rate} which is lower than the requested sample rate of {sampling_rate}. This is not a good idea.')
            audio = torch.nn.functional.interpolate(audio.unsqueeze(0).unsqueeze(1), scale_factor=sampling_rate/lsr, mode='nearest', recompute_scale_factor=False).squeeze()

        # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
        # '2' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
        if torch.any(audio > 2) or not torch.any(audio < 0):
            print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
        audio.clip_(-1, 1)
        return audio.unsqueeze(0)


class WavfileDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        path = opt['path']
        cache_path = opt['cache_path']  # Will fail when multiple paths specified, must be specified in this case.
        if not isinstance(path, list):
            path = [path]
        if os.path.exists(cache_path):
            self.audiopaths = torch.load(cache_path)
        else:
            print("Building cache..")
            self.audiopaths = []
            for p in path:
                self.audiopaths.extend(find_files_of_type('img', p, qualifier=is_wav_file)[0])
            torch.save(self.audiopaths, cache_path)

        # Parse options
        self.sampling_rate = opt_get(opt, ['sampling_rate'], 24000)
        self.pad_to = opt_get(opt, ['pad_to_seconds'], None)
        if self.pad_to is not None:
            self.pad_to *= self.sampling_rate
        self.pad_to = opt_get(opt, ['pad_to_samples'], self.pad_to)

        self.augment = opt_get(opt, ['do_augmentation'], False)
        if self.augment:
            # The "window size" for the clips produced in seconds.
            self.window = 2 * self.sampling_rate
            self.augmentor = WavAugmentor()

    def get_audio_for_index(self, index):
        audiopath = self.audiopaths[index]
        audio = load_audio_from_wav(audiopath, self.sampling_rate)
        return audio, audiopath

    def __getitem__(self, index):
        success = False
        # This "success" thing is a hack: This dataset is randomly failing for no apparent good reason and I don't know why.
        # Symptoms are it complaining about being unable to read a nonsensical filename that is clearly corrupted. Memory corruption? I don't know..
        while not success:
            try:
                # Split audio_norm into two tensors of equal size.
                audio_norm, filename = self.get_audio_for_index(index)
                success = True
            except:
                print(f"Failed to load {index} {self.audiopaths[index]}")

        if self.augment:
            if audio_norm.shape[1] < self.window * 2:
                # Try next index. This adds a bit of bias and ideally we'd filter the dataset rather than do this.
                return self[(index + 1) % len(self)]
            j = random.randint(0, audio_norm.shape[1] - self.window)
            clip1 = audio_norm[:, j:j+self.window]
            if self.augment:
                clip1 = self.augmentor.augment(clip1, self.sampling_rate)
            j = random.randint(0, audio_norm.shape[1]-self.window)
            clip2 = audio_norm[:, j:j+self.window]
            if self.augment:
                clip2 = self.augmentor.augment(clip2, self.sampling_rate)

        # This is required when training to make sure all clips align.
        if self.pad_to is not None:
            if audio_norm.shape[-1] <= self.pad_to:
                audio_norm = torch.nn.functional.pad(audio_norm, (0, self.pad_to - audio_norm.shape[-1]))
            else:
                gap = audio_norm.shape[-1] - self.pad_to
                start = random.randint(0, gap-1)
                audio_norm = audio_norm[:, start:start+self.pad_to]

        output = {
            'clip': audio_norm,
            'path': filename,
        }
        if self.augment:
            output.update({
                'clip1': clip1[0, :].unsqueeze(0),
                'clip2': clip2[0, :].unsqueeze(0),
            })
        return output

    def __len__(self):
        return len(self.audiopaths)


if __name__ == '__main__':
    params = {
        'mode': 'wavfile_clips',
        'path': ['E:\\audio\\books-split', 'E:\\audio\\LibriTTS\\train-clean-360', 'D:\\data\\audio\\podcasts-split'],
        'cache_path': 'E:\\audio\\clips-cache.pth',
        'sampling_rate': 22050,
        'pad_to_seconds': 5,
        'phase': 'train',
        'n_workers': 0,
        'batch_size': 16,
        'do_augmentation': False
    }
    from data import create_dataset, create_dataloader, util

    ds = create_dataset(params)
    dl = create_dataloader(ds, params)
    i = 0
    for b in tqdm(dl):
        for b_ in range(16):
            pass
            #torchaudio.save(f'{i}_clip1_{b_}.wav', b['clip1'][b_], ds.sampling_rate)
            #torchaudio.save(f'{i}_clip2_{b_}.wav', b['clip2'][b_], ds.sampling_rate)
            #i += 1
