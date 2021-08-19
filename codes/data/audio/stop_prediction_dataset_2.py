import os
import pathlib
import random

from munch import munchify
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

from data.audio.nv_tacotron_dataset import save_mel_buffer_to_file
from models.tacotron2 import hparams
from models.tacotron2.layers import TacotronSTFT
from models.tacotron2.taco_utils import load_wav_to_torch
from utils.util import opt_get


# A dataset that consumes the result from the script `produce_libri_stretched_dataset`, which itself is a combined
# set of clips from the librivox corpus of equal length with the sentence alignment labeled.
class StopPredictionDataset(Dataset):
    def __init__(self, opt):
        path = opt['path']
        label_compaction = opt_get(opt, ['label_compaction'], 1)
        hp = munchify(hparams.create_hparams())
        cache_path = os.path.join(path, 'cache.pth')
        if os.path.exists(cache_path):
            self.files = torch.load(cache_path)
        else:
            print("Building cache..")
            self.files = list(pathlib.Path(path).glob('*.wav'))
            torch.save(self.files, cache_path)
        self.sampling_rate = 22050  # Fixed since the underlying data is also fixed at this SR.
        self.mel_length = 2000
        self.stft = TacotronSTFT(
            hp.filter_length, hp.hop_length, hp.win_length,
            hp.n_mel_channels, hp.sampling_rate, hp.mel_fmin,
            hp.mel_fmax)
        self.label_compaction = label_compaction

    def __getitem__(self, index):
        audio, _ = load_wav_to_torch(self.files[index])
        starts, ends = torch.load(str(self.files[index]).replace('.wav', '_se.pth'))

        if audio.std() > 1:
            print(f"Something is very wrong with the given audio. std_dev={audio.std()}. file={self.files[index]}")
            return None
        audio.clip_(-1, 1)
        mels = self.stft.mel_spectrogram(audio.unsqueeze(0))[:, :, :self.mel_length].squeeze(0)

        # Form labels.
        labels_start = torch.zeros((2000 // self.label_compaction,), dtype=torch.long)
        for s in starts:
            # Mel compaction operates at a ratio of 1/256, the dataset also allows further compaction.
            s = s // (256 * self.label_compaction)
            if s >= 2000//self.label_compaction:
                continue
            labels_start[s] = 1
        labels_end = torch.zeros((2000 // self.label_compaction,), dtype=torch.long)
        for e in ends:
            e = e // (256 * self.label_compaction)
            if e >= 2000//self.label_compaction:
                continue
            labels_end[e] = 1

        return {
            'mels': mels,
            'labels_start': labels_start,
            'labels_end': labels_end,
        }


    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    opt = {
        'path': 'D:\\data\\audio\\libritts\\stop_dataset',
        'label_compaction': 4,
    }
    ds = StopPredictionDataset(opt)
    j = 0
    for i in tqdm(range(100)):
        b = ds[random.randint(0, len(ds))]
        start_indices = torch.nonzero(b['labels_start']).squeeze(1)
        end_indices = torch.nonzero(b['labels_end']).squeeze(1)
        assert len(end_indices) <= len(start_indices)  # There should always be more START tokens then END tokens.
        for i in range(len(end_indices)):
            s = start_indices[i].item()*4
            e = end_indices[i].item()*4
            m = b['mels'][:, s:e]
            save_mel_buffer_to_file(m, f'{j}.npy')
            j += 1