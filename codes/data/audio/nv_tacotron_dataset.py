import os
import random

import audio2numpy
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

import models.tacotron2.layers as layers
from models.tacotron2.taco_utils import load_wav_to_torch, load_filepaths_and_text

from models.tacotron2.text import text_to_sequence
from utils.util import opt_get


def load_mozilla_cv(filename):
    with open(filename, encoding='utf-8') as f:
        components = [line.strip().split('\t') for line in f][1:]  # First line is the header
        filepaths_and_text = [[f'clips/{component[1]}', component[2]] for component in components]
    return filepaths_and_text


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, hparams):
        self.path = os.path.dirname(hparams['path'])
        fetcher_mode = opt_get(hparams, ['fetcher_mode'], 'lj')
        fetcher_fn = None
        if fetcher_mode == 'lj':
            fetcher_fn = load_filepaths_and_text
        elif fetcher_mode == 'mozilla_cv':
            fetcher_fn = load_mozilla_cv
        else:
            raise NotImplementedError()
        self.audiopaths_and_text = fetcher_fn(hparams['path'])
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.return_wavs = opt_get(hparams, ['return_wavs'], False)
        self.input_sample_rate = opt_get(hparams, ['input_sample_rate'], self.sampling_rate)
        assert not (self.load_mel_from_disk and self.return_wavs)
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        audiopath = os.path.join(self.path, audiopath)
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel, audiopath_and_text[0])

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            if filename.endswith('.wav'):
                audio, sampling_rate = load_wav_to_torch(filename)
                audio = audio / self.max_wav_value
            else:
                audio, sampling_rate = audio2numpy.audio_from_file(filename)
                audio = torch.tensor(audio)

            if sampling_rate != self.input_sample_rate:
                assert sampling_rate > self.input_sample_rate  # Upsampling is not a great idea.
                audio = torch.nn.functional.interpolate(audio.unsqueeze(0).unsqueeze(1), scale_factor=self.input_sample_rate/sampling_rate, mode='area')
                audio = (audio.squeeze().clip(-1,1)+1)/2
            audio_norm = audio.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            if self.input_sample_rate != self.sampling_rate:
                ratio = self.sampling_rate / self.input_sample_rate
                audio_norm = torch.nn.functional.interpolate(audio_norm.unsqueeze(0), scale_factor=ratio, mode='area').squeeze(0)
            if self.return_wavs:
                melspec = audio_norm
            else:
                melspec = self.stft.mel_spectrogram(audio_norm)
                melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, filename]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        filenames = []
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
            filenames.append(batch[ids_sorted_decreasing[i]][2])

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)


        return {
            'padded_text': text_padded,
            'input_lengths': input_lengths,
            'padded_mel': mel_padded,
            'padded_gate': gate_padded,
            'output_lengths': output_lengths,
            'filenames': filenames
        }


if __name__ == '__main__':
    params = {
        'mode': 'nv_tacotron',
        'path': 'E:\\audio\\MozillaCommonVoice\\en\\test.tsv',
        'phase': 'train',
        'n_workers': 0,
        'batch_size': 32,
        'fetcher_mode': 'mozilla_cv',
        #'return_wavs': True,
        #'input_sample_rate': 22050,
        #'sampling_rate': 8000
    }
    from data import create_dataset, create_dataloader

    ds, c = create_dataset(params, return_collate=True)
    dl = create_dataloader(ds, params, collate_fn=c)
    i = 0
    m = None
    for i, b in tqdm(enumerate(dl)):
        pm = b['padded_mel']
        pm = torch.nn.functional.pad(pm, (0, 800-pm.shape[-1]))
        m = pm if m is None else torch.cat([m, pm], dim=0)
        print(m.mean(), m.std())
