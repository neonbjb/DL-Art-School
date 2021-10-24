import os
import pathlib
import random
import sys
from warnings import warn

import torch
import torch.utils.data
import torch.nn.functional as F
import torchaudio
from audio2numpy import open_audio
from tqdm import tqdm

from data.audio.wav_aug import WavAugmentor
from data.util import find_files_of_type, is_wav_file, is_audio_file, load_paths_from_cache
from models.tacotron2.taco_utils import load_wav_to_torch
from utils.util import opt_get


def load_audio(audiopath, sampling_rate):
    if audiopath[-4:] == '.wav':
        audio, lsr = load_wav_to_torch(audiopath)
    else:
        audio, lsr = open_audio(audiopath)
        audio = torch.FloatTensor(audio)

    # Remove any channel data.
    if len(audio.shape) > 1:
        if audio.shape[0] < 5:
            audio = audio[0]
        else:
            assert audio.shape[1] < 5
            audio = audio[:, 0]

    if lsr != sampling_rate:
        #if lsr < sampling_rate:
        #    warn(f'{audiopath} has a sample rate of {sampling_rate} which is lower than the requested sample rate of {sampling_rate}. This is not a good idea.')
        audio = torch.nn.functional.interpolate(audio.unsqueeze(0).unsqueeze(1), scale_factor=sampling_rate/lsr, mode='nearest', recompute_scale_factor=False).squeeze()

    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '2' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1, 1)

    return audio.unsqueeze(0)


class UnsupervisedAudioDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        path = opt['path']
        cache_path = opt['cache_path']  # Will fail when multiple paths specified, must be specified in this case.
        self.audiopaths = load_paths_from_cache(path, cache_path)

        # Parse options
        self.sampling_rate = opt_get(opt, ['sampling_rate'], 22050)
        self.pad_to = opt_get(opt, ['pad_to_seconds'], None)
        if self.pad_to is not None:
            self.pad_to *= self.sampling_rate
        self.pad_to = opt_get(opt, ['pad_to_samples'], self.pad_to)

        self.extra_samples = opt_get(opt, ['extra_samples'], 0)
        self.extra_sample_len = opt_get(opt, ['extra_sample_length'], 2)
        self.extra_sample_len *= self.sampling_rate

    def get_audio_for_index(self, index):
        audiopath = self.audiopaths[index]
        audio = load_audio(audiopath, self.sampling_rate)
        return audio, audiopath

    def get_related_audio_for_index(self, index):
        if self.extra_samples <= 0:
            return None, 0
        audiopath = self.audiopaths[index]
        related_files = find_files_of_type('img', os.path.dirname(audiopath), qualifier=is_audio_file)[0]
        assert audiopath in related_files
        assert len(related_files) < 50000  # Sanity check to ensure we aren't loading "related files" that aren't actually related.
        if len(related_files) == 0:
            print(f"No related files for {audiopath}")
        related_files.remove(audiopath)
        related_clips = []
        random.shuffle(related_clips)
        i = 0
        for related_file in related_files:
            rel_clip = load_audio(related_file, self.sampling_rate)
            gap = rel_clip.shape[-1] - self.extra_sample_len
            if gap < 0:
                rel_clip = F.pad(rel_clip, pad=(0, abs(gap)))
            elif gap > 0:
                rand_start = random.randint(0, gap)
                rel_clip = rel_clip[:, rand_start:rand_start+self.extra_sample_len]
            related_clips.append(rel_clip)
            i += 1
            if i >= self.extra_samples:
                break
        actual_extra_samples = i
        while i < self.extra_samples:
            related_clips.append(torch.zeros(1, self.extra_sample_len))
            i += 1
        return torch.stack(related_clips, dim=0), actual_extra_samples

    def __getitem__(self, index):
        try:
            # Split audio_norm into two tensors of equal size.
            audio_norm, filename = self.get_audio_for_index(index)
            alt_files, actual_samples = self.get_related_audio_for_index(index)
        except:
            print(f"Error loading audio for file {self.audiopaths[index]} {sys.exc_info()}")
            return self[index+1]

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
        if self.extra_samples > 0:
            output['alt_clips'] = alt_files
            output['num_alt_clips'] = actual_samples
        return output

    def __len__(self):
        return len(self.audiopaths)


if __name__ == '__main__':
    params = {
        'mode': 'unsupervised_audio',
        'path': ['\\\\192.168.5.3\\rtx3080_audio_y\\split\\books2', '\\\\192.168.5.3\\rtx3080_audio\\split\\books1', '\\\\192.168.5.3\\rtx3080_audio\\split\\cleaned-2'],
        'cache_path': 'E:\\audio\\remote-cache2.pth',
        'sampling_rate': 22050,
        'pad_to_samples': 40960,
        'phase': 'train',
        'n_workers': 1,
        'batch_size': 16,
        'extra_samples': 4,
    }
    from data import create_dataset, create_dataloader, util

    ds = create_dataset(params)
    dl = create_dataloader(ds, params)
    i = 0
    for b in tqdm(dl):
        for b_ in range(b['clip'].shape[0]):
            #pass
            torchaudio.save(f'{i}_clip_{b_}.wav', b['clip'][b_], ds.sampling_rate)
            i += 1
