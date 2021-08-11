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


class WavfileDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        cache_path = opt_get(opt, ['cache_path'], os.path.join(self.path, 'cache.pth'))  # Will fail when multiple paths specified, must be specified in this case.
        self.path = os.path.dirname(opt['path'])
        if not isinstance(self.path, list):
            self.path = [self.path]
        if os.path.exists(cache_path):
            self.audiopaths = torch.load(cache_path)
        else:
            print("Building cache..")
            self.audiopaths = []
            for p in self.path:
                self.audiopaths.extend(find_files_of_type('img', p, qualifier=is_wav_file)[0])
            torch.save(self.audiopaths, cache_path)

        # Parse options
        self.sampling_rate = opt_get(opt, ['sampling_rate'], 24000)
        self.augment = opt_get(opt, ['do_augmentation'], False)
        self.max_wav_value = 32768.0

        self.window = 2 * self.sampling_rate
        if self.augment:
            self.augmentor = WavAugmentor()

    def get_audio_for_index(self, index):
        audiopath = self.audiopaths[index]
        filename = os.path.join(self.path, audiopath)
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"Input sampling rate does not match specified rate {self.sampling_rate}")
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        return audio_norm, audiopath

    def __getitem__(self, index):
        clip1, clip2 = None, None

        while clip1 is None and clip2 is None:
            # Split audio_norm into two tensors of equal size.
            audio_norm, filename = self.get_audio_for_index(index)
            if audio_norm.shape[1] < self.window * 2:
                # Try next index. This adds a bit of bias and ideally we'd filter the dataset rather than do this.
                index = (index + 1) % len(self)
                continue
            j = random.randint(0, audio_norm.shape[1] - self.window)
            clip1 = audio_norm[:, j:j+self.window]
            if self.augment:
                clip1 = self.augmentor.augment(clip1, self.sampling_rate)
            j = random.randint(0, audio_norm.shape[1]-self.window)
            clip2 = audio_norm[:, j:j+self.window]
            if self.augment:
                clip2 = self.augmentor.augment(clip2, self.sampling_rate)

        return {
            'clip1': clip1[0, :].unsqueeze(0),
            'clip2': clip2[0, :].unsqueeze(0),
            'path': filename,
        }

    def __len__(self):
        return len(self.audiopaths)


if __name__ == '__main__':
    params = {
        'mode': 'wavfile_clips',
        'path': 'E:\\audio\\LibriTTS\\train-other-500',
        'phase': 'train',
        'n_workers': 0,
        'batch_size': 16,
        'do_augmentation': True,
    }
    from data import create_dataset, create_dataloader, util

    ds = create_dataset(params, return_collate=True)
    dl = create_dataloader(ds, params, collate_fn=c)
    i = 0
    for b in tqdm(dl):
        for b_ in range(16):
            torchaudio.save(f'{i}_clip1_{b_}.wav', b['clip1'][b_], ds.sampling_rate)
            torchaudio.save(f'{i}_clip2_{b_}.wav', b['clip2'][b_], ds.sampling_rate)
            i += 1
