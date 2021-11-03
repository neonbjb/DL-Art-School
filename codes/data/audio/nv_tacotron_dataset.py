import os
import random

import audio2numpy
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

import models.tacotron2.layers as layers
from data.audio.unsupervised_audio_dataset import load_audio
from models.tacotron2.taco_utils import load_wav_to_torch, load_filepaths_and_text

from models.tacotron2.text import text_to_sequence
from utils.util import opt_get


def load_mozilla_cv(filename):
    with open(filename, encoding='utf-8') as f:
        components = [line.strip().split('\t') for line in f][1:]  # First line is the header
        base = os.path.dirname(filename)
        filepaths_and_text = [[os.path.join(base, f'clips/{component[1]}'), component[2]] for component in components]
    return filepaths_and_text


def load_voxpopuli(filename):
    with open(filename, encoding='utf-8') as f:
        lines = [line.strip().split('\t') for line in f][1:]  # First line is the header
        base = os.path.dirname(filename)
        filepaths_and_text = []
        for line in lines:
            if len(line) == 0:
                continue
            file, raw_text, norm_text, speaker_id, split, gender = line
            year = file[:4]
            filepaths_and_text.append([os.path.join(base, year, f'{file}.ogg.wav'), raw_text])
    return filepaths_and_text


class TextWavLoader(torch.utils.data.Dataset):
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
            elif fm == 'mozilla_cv':
                fetcher_fn = load_mozilla_cv
            elif fm == 'voxpopuli':
                fetcher_fn = load_voxpopuli
            else:
                raise NotImplementedError()
            self.audiopaths_and_text.extend(fetcher_fn(p))
        self.text_cleaners = hparams.text_cleaners
        self.sample_rate = hparams.sample_rate
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)
        self.max_wav_len = opt_get(hparams, ['max_wav_length'], None)
        self.max_text_len = opt_get(hparams, ['max_text_length'], None)
        # If needs_collate=False, all outputs will be aligned and padded at maximum length.
        self.needs_collate = opt_get(hparams, ['needs_collate'], True)
        if not self.needs_collate:
            assert self.max_wav_len is not None and self.max_text_len is not None

    def get_wav_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text_seq = self.get_text(text)
        wav = load_audio(audiopath, self.sample_rate)
        return (text_seq, wav, text, audiopath_and_text[0])

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        try:
            tseq, wav, text, path = self.get_wav_text_pair(self.audiopaths_and_text[index])
        except:
            print(f"error loadding {self.audiopaths_and_text[index][0]")
            return self[index+1]
        if wav is None or \
            (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len) or \
            (self.max_text_len is not None and tseq.shape[0] > self.max_text_len):
            # Basically, this audio file is nonexistent or too long to be supported by the dataset.
            # It's hard to handle this situation properly. Best bet is to return the a random valid token and skew the dataset somewhat as a result.
            #if wav is not None:
            #    print(f"Exception {index} wav_len:{wav.shape[-1]} text_len:{tseq.shape[0]} fname: {path}")
            rv = random.randint(0,len(self)-1)
            return self[rv]
        orig_output = wav.shape[-1]
        orig_text_len = tseq.shape[0]
        if not self.needs_collate:
            if wav.shape[-1] != self.max_wav_len:
                wav = F.pad(wav, (0, self.max_wav_len - wav.shape[-1]))
            if tseq.shape[0] != self.max_text_len:
                tseq = F.pad(tseq, (0, self.max_text_len - tseq.shape[0]))
            return {
                'real_text': text,
                'padded_text': tseq,
                'input_lengths': torch.tensor(orig_text_len, dtype=torch.long),
                'wav': wav,
                'output_lengths': torch.tensor(orig_output, dtype=torch.long),
                'filenames': path
            }
        return tseq, wav, path, text

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __call__(self, batch):
        """Collate's training batch from normalized text and wav
        PARAMS
        ------
        batch: [text_normalized, wav, filename, text]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        filenames = []
        real_text = []
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
            filenames.append(batch[ids_sorted_decreasing[i]][2])
            real_text.append(batch[ids_sorted_decreasing[i]][3])

        # Right zero-pad wav
        num_wavs = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # include mel padded and gate padded
        wav_padded = torch.FloatTensor(len(batch), num_wavs, max_target_len)
        wav_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            wav = batch[ids_sorted_decreasing[i]][1]
            wav_padded[i, :, :wav.size(1)] = wav
            output_lengths[i] = wav.size(1)

        return {
            'padded_text': text_padded,
            'input_lengths': input_lengths,
            'wav': wav_padded,
            'output_lengths': output_lengths,
            'filenames': filenames,
            'real_text': real_text,
        }


if __name__ == '__main__':
    batch_sz = 32
    params = {
        'mode': 'nv_tacotron',
        'path': 'E:\\audio\\MozillaCommonVoice\\en\\test.tsv',
        'phase': 'train',
        'n_workers': 0,
        'batch_size': batch_sz,
        'fetcher_mode': 'mozilla_cv',
        'needs_collate': True,
        #'max_wav_length': 256000,
        #'max_text_length': 200,
        'sample_rate': 22050,
    }
    from data import create_dataset, create_dataloader

    ds, c = create_dataset(params, return_collate=True)
    dl = create_dataloader(ds, params, collate_fn=c)
    i = 0
    m = None
    for k in range(1000):
        for i, b in tqdm(enumerate(dl)):
            w = b['wav']
            for ib in range(batch_sz):
                print(f'{i} {ib} {b["real_text"][ib]}')
                torchaudio.save(f'{i}_clip_{ib}.wav', b['wav'][ib], ds.sample_rate)
