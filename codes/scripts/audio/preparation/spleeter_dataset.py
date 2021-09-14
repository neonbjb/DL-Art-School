import torch
import torch.nn as nn
from spleeter.audio.adapter import AudioAdapter
from torch.utils.data import Dataset

from data.util import find_audio_files


class SpleeterDataset(Dataset):
    def __init__(self, src_dir, sample_rate=22050, max_duration=20, skip=0):
        self.files = find_audio_files(src_dir, include_nonwav=True)
        if skip > 0:
            self.files = self.files[skip:]
        self.audio_loader = AudioAdapter.default()
        self.sample_rate = sample_rate
        self.max_duration = max_duration

    def __getitem__(self, item):
        file = self.files[item]
        try:
            wave, sample_rate = self.audio_loader.load(file, sample_rate=self.sample_rate)
            assert sample_rate == self.sample_rate
            wave = wave[:,0]  # strip off channels
            wave = torch.tensor(wave)
        except:
            wave = torch.zeros(self.sample_rate * self.max_duration)
            print(f"Error with {file}")
        original_duration = wave.shape[0]
        padding_needed = self.sample_rate * self.max_duration - original_duration
        if padding_needed > 0:
            wave = nn.functional.pad(wave, (0, padding_needed))
        return {
            'path': file,
            'wave': wave,
            'duration': original_duration,
        }

    def __len__(self):
        return len(self.files)