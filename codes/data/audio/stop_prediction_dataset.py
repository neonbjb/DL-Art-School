import os
import pathlib
import random

import audio2numpy
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm

import models.tacotron2.layers as layers
from data.audio.nv_tacotron_dataset import load_mozilla_cv, load_voxpopuli
from models.tacotron2.taco_utils import load_wav_to_torch, load_filepaths_and_text

from models.tacotron2.text import text_to_sequence
from utils.util import opt_get


def get_similar_files_libritts(filename):
    filedir = os.path.dirname(filename)
    return list(pathlib.Path(filedir).glob('*.wav'))


class StopPredictionDataset(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, hparams):
        self.path = hparams['path']
        if not isinstance(self.path, list):
            self.path = [self.path]

        fetcher_mode = opt_get(hparams, ['fetcher_mode'], 'lj')
        if not isinstance(fetcher_mode, list):
            fetcher_mode = [fetcher_mode]
        assert len(self.path) == len(fetcher_mode)

        self.audiopaths_and_text = []
        for p, fm in zip(self.path, fetcher_mode):
            if fm == 'lj' or fm == 'libritts':
                fetcher_fn = load_filepaths_and_text
                self.get_similar_files = get_similar_files_libritts
            elif fm == 'voxpopuli':
                fetcher_fn = load_voxpopuli
                self.get_similar_files = None  # TODO: Fix.
            else:
                raise NotImplementedError()
            self.audiopaths_and_text.extend(fetcher_fn(p))
        self.sampling_rate = hparams.sampling_rate
        self.input_sample_rate = opt_get(hparams, ['input_sample_rate'], self.sampling_rate)
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)
        self.max_mel_len = opt_get(hparams, ['max_mel_length'], None)
        self.max_text_len = opt_get(hparams, ['max_text_length'], None)

    def get_mel(self, filename):
        filename = str(filename)
        if filename.endswith('.wav'):
            audio, sampling_rate = load_wav_to_torch(filename)
        else:
            audio, sampling_rate = audio2numpy.audio_from_file(filename)
            audio = torch.tensor(audio)

        if sampling_rate != self.input_sample_rate:
            if sampling_rate < self.input_sample_rate:
                print(f'{filename} has a sample rate of {sampling_rate} which is lower than the requested sample rate of {self.input_sample_rate}. This is not a good idea.')
            audio_norm = torch.nn.functional.interpolate(audio.unsqueeze(0).unsqueeze(1), scale_factor=self.input_sample_rate/sampling_rate, mode='nearest', recompute_scale_factor=False).squeeze()
        else:
            audio_norm = audio
        if audio_norm.std() > 1:
            print(f"Something is very wrong with the given audio. std_dev={audio_norm.std()}. file={filename}")
            return None
        audio_norm.clip_(-1, 1)
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        if self.input_sample_rate != self.sampling_rate:
            ratio = self.sampling_rate / self.input_sample_rate
            audio_norm = torch.nn.functional.interpolate(audio_norm.unsqueeze(0), scale_factor=ratio, mode='area').squeeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return melspec

    def __getitem__(self, index):
        path = self.audiopaths_and_text[index][0]
        similar_files = self.get_similar_files(path)
        mel = self.get_mel(path)
        terms = torch.zeros(mel.shape[1])
        terms[-1] = 1
        while mel.shape[-1] < self.max_mel_len:
            another_file = random.choice(similar_files)
            another_mel = self.get_mel(another_file)
            oterms = torch.zeros(another_mel.shape[1])
            oterms[-1] = 1
            mel = torch.cat([mel, another_mel], dim=-1)
            terms = torch.cat([terms, oterms], dim=-1)
        mel = mel[:, :self.max_mel_len]
        terms = terms[:self.max_mel_len]


        return {
            'padded_mel': mel,
            'termination_mask': terms,
        }

    def __len__(self):
        return len(self.audiopaths_and_text)


if __name__ == '__main__':
    params = {
        'mode': 'stop_prediction',
        'path': 'E:\\audio\\LibriTTS\\train-clean-360_list.txt',
        'phase': 'train',
        'n_workers': 0,
        'batch_size': 16,
        'fetcher_mode': 'libritts',
        'max_mel_length': 800,
        #'return_wavs': True,
        #'input_sample_rate': 22050,
        #'sampling_rate': 8000
    }
    from data import create_dataset, create_dataloader

    ds, c = create_dataset(params, return_collate=True)
    dl = create_dataloader(ds, params, collate_fn=c, shuffle=True)
    i = 0
    m = None
    for k in range(1000):
        for i, b in tqdm(enumerate(dl)):
            continue
            pm = b['padded_mel']
            pm = torch.nn.functional.pad(pm, (0, 800-pm.shape[-1]))
            m = pm if m is None else torch.cat([m, pm], dim=0)
            print(m.mean(), m.std())