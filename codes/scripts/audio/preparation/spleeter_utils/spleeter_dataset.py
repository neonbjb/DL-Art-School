from math import ceil

import numpy as np

from spleeter.audio.adapter import AudioAdapter
from torch.utils.data import Dataset

from data.util import find_audio_files


class SpleeterDataset(Dataset):
    def __init__(self, src_dir, batch_sz, max_duration, sample_rate=22050, partition=None, partition_size=None, resume=None):
        self.batch_sz = batch_sz
        self.max_duration = max_duration
        self.files = find_audio_files(src_dir, include_nonwav=True)
        self.sample_rate = sample_rate

        # Partition files if needed.
        if partition_size is not None:
            psz = int(partition_size)
            prt = int(partition)
            self.files = self.files[prt * psz:(prt + 1) * psz]

        # Find the resume point and carry on from there.
        if resume is not None:
            for i, f in enumerate(self.files):
                if resume in f:
                    break
            assert i < len(self.files)
            self.files = self.files[i:]
        self.loader = AudioAdapter.default()

    def __len__(self):
        return ceil(len(self.files) / self.batch_sz)

    def __getitem__(self, item):
        item = item * self.batch_sz
        wavs = None
        files = []
        ends = []
        for k in range(self.batch_sz):
            ind = k+item
            if ind >= len(self.files):
                break

            #try:
            wav, sr = self.loader.load(self.files[ind], sample_rate=self.sample_rate)
            assert sr == 22050
            # Get rid of all channels except one.
            if wav.shape[1] > 1:
                wav = wav[:, 0]

            if wavs is None:
                wavs = wav
            else:
                wavs = np.concatenate([wavs, wav])
            ends.append(wavs.shape[0])
            files.append(self.files[ind])
            #except:
            #    print(f'Error loading {self.files[ind]}')
        return {
            'audio': wavs,
            'files': files,
            'ends': ends
        }