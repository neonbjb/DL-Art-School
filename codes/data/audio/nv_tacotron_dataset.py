import os
import random

import audio2numpy
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm

import models.tacotron2.layers as layers
from models.tacotron2.taco_utils import load_wav_to_torch, load_filepaths_and_text

from models.tacotron2.text import text_to_sequence
from utils.util import opt_get


def load_mozilla_cv(filename):
    with open(filename, encoding='utf-8') as f:
        components = [line.strip().split('\t') for line in f][1:]  # First line is the header
        base = os.path.dirname(filename)
        filepaths_and_text = [[os.path.join(base, f'clips/{component[1]}'), component[2]] for component in components]
    return filepaths_and_text


class TextMelLoader(torch.utils.data.Dataset):
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
            elif fm == 'mozilla_cv':
                fetcher_fn = load_mozilla_cv
            else:
                raise NotImplementedError()
            self.audiopaths_and_text.extend(fetcher_fn(p))
        self.text_cleaners = hparams.text_cleaners
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = opt_get(hparams, ['load_mel_from_disk'], False)
        self.return_wavs = opt_get(hparams, ['return_wavs'], False)
        self.input_sample_rate = opt_get(hparams, ['input_sample_rate'], self.sampling_rate)
        assert not (self.load_mel_from_disk and self.return_wavs)
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)
        self.max_mel_len = opt_get(hparams, ['max_mel_length'], None)
        self.max_text_len = opt_get(hparams, ['max_text_length'], None)
        # If needs_collate=False, all outputs will be aligned and padded at maximum length.
        self.needs_collate = opt_get(hparams, ['needs_collate'], True)
        if not self.needs_collate:
            assert self.max_mel_len is not None and self.max_text_len is not None

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text_seq = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text_seq, mel, text, audiopath_and_text[0])

    def get_mel(self, filename):
        if self.load_mel_from_disk and os.path.exists(f'{filename}_mel.npy'):
            melspec = torch.from_numpy(np.load(f'{filename}_mel.npy'))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(melspec.size(0), self.stft.n_mel_channels))
        else:
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
            if self.return_wavs:
                melspec = audio_norm
            else:
                melspec = self.stft.mel_spectrogram(audio_norm)
                melspec = torch.squeeze(melspec, 0)

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        tseq, mel, text, path = self.get_mel_text_pair(self.audiopaths_and_text[index])
        if mel is None or \
            (self.max_mel_len is not None and mel.shape[-1] > self.max_mel_len) or \
            (self.max_text_len is not None and tseq.shape[0] > self.max_text_len):
            #if mel is not None:
            #    print(f"Exception {index} mel_len:{mel.shape[-1]} text_len:{tseq.shape[0]} fname: {path}")
            # It's hard to handle this situation properly. Best bet is to return the a random valid token and skew the dataset somewhat as a result.
            rv = random.randint(0,len(self)-1)
            return self[rv]
        orig_output = mel.shape[-1]
        orig_text_len = tseq.shape[0]
        if not self.needs_collate:
            if mel.shape[-1] != self.max_mel_len:
                mel = F.pad(mel, (0, self.max_mel_len - mel.shape[-1]))
            if tseq.shape[0] != self.max_text_len:
                tseq = F.pad(tseq, (0, self.max_text_len - tseq.shape[0]))
            return {
                'real_text': text,
                'padded_text': tseq,
                'input_lengths': torch.tensor(orig_text_len, dtype=torch.long),
                'padded_mel': mel,
                'output_lengths': torch.tensor(orig_output, dtype=torch.long),
                'filenames': path
            }
        return tseq, mel, path, text

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
        real_text = []
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
            filenames.append(batch[ids_sorted_decreasing[i]][2])
            real_text.append(batch[ids_sorted_decreasing[i]][3])

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
            'filenames': filenames,
            'real_text': real_text,
        }


def save_mel_buffer_to_file(mel, path):
    np.save(path, mel.numpy())


def dump_mels_to_disk():
    params = {
        'mode': 'nv_tacotron',
        'path': ['E:\\audio\\MozillaCommonVoice\\en\\test.tsv', 'E:\\audio\\LibriTTS\\train-other-500_list.txt'],
        'fetcher_mode': ['mozilla_cv', 'libritts'],
        'phase': 'train',
        'n_workers': 0,
        'batch_size': 1,
        'needs_collate': True,
        'max_mel_length': 1000,
        'max_text_length': 200,
        #'return_wavs': True,
        #'input_sample_rate': 22050,
        #'sampling_rate': 8000
    }
    output_path = 'D:\\dlas\\results\\mozcv_mels'
    os.makedirs(os.path.join(output_path, 'clips'), exist_ok=True)
    from data import create_dataset, create_dataloader
    ds, c = create_dataset(params, return_collate=True)
    dl = create_dataloader(ds, params, collate_fn=c)
    for i, b in tqdm(enumerate(dl)):
        mels = b['padded_mel']
        fnames = b['filenames']
        for j, fname in enumerate(fnames):
            save_mel_buffer_to_file(mels[j], f'{os.path.join(output_path, fname)}_mel.npy')


if __name__ == '__main__':
    dump_mels_to_disk()
    '''
    params = {
        'mode': 'nv_tacotron',
        'path': 'E:\\audio\\MozillaCommonVoice\\en\\train.tsv',
        'phase': 'train',
        'n_workers': 12,
        'batch_size': 32,
        'fetcher_mode': 'mozilla_cv',
        'needs_collate': False,
        'max_mel_length': 800,
        'max_text_length': 200,
        #'return_wavs': True,
        #'input_sample_rate': 22050,
        #'sampling_rate': 8000
    }
    from data import create_dataset, create_dataloader

    ds, c = create_dataset(params, return_collate=True)
    dl = create_dataloader(ds, params, collate_fn=c)
    i = 0
    m = None
    for k in range(1000):
        for i, b in tqdm(enumerate(dl)):
            continue
            pm = b['padded_mel']
            pm = torch.nn.functional.pad(pm, (0, 800-pm.shape[-1]))
            m = pm if m is None else torch.cat([m, pm], dim=0)
            print(m.mean(), m.std())
    '''