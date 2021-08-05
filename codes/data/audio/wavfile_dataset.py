import os
import random

import torch
import torch.utils.data
from tqdm import tqdm

from data.util import get_image_paths, is_wav_file
from models.tacotron2.taco_utils import load_wav_to_torch


class WavfileDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        self.path = os.path.dirname(opt['path'])
        cache_path = os.path.join(self.path, 'cache.pth')
        if os.path.exists(cache_path):
            self.audiopaths = torch.load(cache_path)
        else:
            print("Building cache..")
            self.audiopaths = get_image_paths('img', opt['path'], qualifier=is_wav_file)[0]
            torch.save(self.audiopaths, cache_path)
        self.max_wav_value = 32768.0
        self.sampling_rate = 24000
        self.window = 2 * self.sampling_rate

    def get_audio_for_index(self, index):
        audiopath = self.audiopaths[index]
        filename = os.path.join(self.path, audiopath)
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"Input sampling rate does not match specified rate {self.sampling_rate}")
        audio_norm = audio / self.max_wav_value
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        return audio_norm, audiopath

    def __getitem__(self, index):
        clip1, clip2 = None, None

        while clip1 is None and clip2 is None:
            # Split audio_norm into two tensors of equal size.
            audio_norm, filename = self.get_audio_for_index(index)
            if audio_norm.shape[0] < self.window * 2:
                # Try next index. This adds a bit of bias and ideally we'd filter the dataset rather than do this.
                index = (index + 1) % len(self)
                continue
            j = random.randint(0, audio_norm.shape[0] - self.window)
            clip1 = audio_norm[j:j+self.window]
            j = random.randint(0, audio_norm.shape[0]-self.window)
            clip2 = audio_norm[j:j+self.window]

        return {
            'clip1': clip1.unsqueeze(0),
            'clip2': clip2.unsqueeze(0),
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
    }
    from data import create_dataset, create_dataloader, util

    ds, c = create_dataset(params, return_collate=True)
    dl = create_dataloader(ds, params, collate_fn=c)
    i = 0
    m = []
    max_text = 0
    max_mel = 0
    for b in tqdm(dl):
        pass
    m=torch.stack(m)
    print(m.mean(), m.std())
