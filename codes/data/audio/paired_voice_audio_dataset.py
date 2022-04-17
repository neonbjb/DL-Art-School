import os
import random
import sys

import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from tqdm import tqdm

from data.audio.unsupervised_audio_dataset import load_audio, load_similar_clips
from models.audio.tts.tacotron2 import load_filepaths_and_text, load_filepaths_and_text_type
from models.audio.tts.tacotron2 import text_to_sequence, sequence_to_text
from utils.util import opt_get


def load_tsv_type(filename, type):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = []
        base = os.path.dirname(filename)
        bad_lines = 0
        for line in f:
            components = line.strip().split('\t')
            if len(components) < 2:
                bad_lines += 1
                if bad_lines > 1000:
                    print(f'{filename} contains 1000+ bad entries. Failing. Sample last entry: {line}')
                    raise ValueError
                continue
            filepaths_and_text.append([os.path.join(base, f'{components[1]}'), components[0]] + [type])
    return filepaths_and_text


def load_tsv(filename):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = []
        base = os.path.dirname(filename)
        bad_lines = 0
        for line in f:
            components = line.strip().split('\t')
            if len(components) < 2:
                bad_lines += 1
                if bad_lines > 1000:
                    print(f'{filename} contains 1000+ bad entries. Failing. Sample last entry: {line}')
                    raise ValueError
                continue
            filepaths_and_text.append([os.path.join(base, f'{components[1]}'), components[0]])
    return filepaths_and_text


def convert_string_list_to_tensor(strlist):
    if strlist.startswith('['):
        strlist = strlist[1:]
    if strlist.endswith(']'):
        strlist = strlist[:-1]
    as_ints = [int(s) for s in strlist.split(', ')]
    return torch.tensor(as_ints)


def load_tsv_aligned_codes_type(filename, type):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = []
        base = os.path.dirname(filename)
        bad_lines = 0
        for line in f:
            components = line.strip().split('\t')
            if len(components) < 3:
                bad_lines += 1
                if bad_lines > 1000:
                    print(f'{filename} contains 1000+ bad entries. Failing. Sample last entry: {line}')
                    raise ValueError
                continue
            filepaths_and_text.append([os.path.join(base, f'{components[1]}'), components[0], convert_string_list_to_tensor(components[2])] + [type])
    return filepaths_and_text


def load_tsv_aligned_codes(filename):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = []
        base = os.path.dirname(filename)
        bad_lines = 0
        for line in f:
            components = line.strip().split('\t')
            if len(components) < 3:
                bad_lines += 1
                if bad_lines > 1000:
                    print(f'{filename} contains 1000+ bad entries. Failing. Sample last entry: {line}')
                    raise ValueError
                continue
            filepaths_and_text.append([os.path.join(base, f'{components[1]}'), components[0], convert_string_list_to_tensor(components[2])])
    return filepaths_and_text


def load_mozilla_cv(filename, type):
    with open(filename, encoding='utf-8') as f:
        components = [line.strip().split('\t') for line in f][1:]  # First line is the header
        base = os.path.dirname(filename)
        filepaths_and_text = [[os.path.join(base, f'clips/{component[1]}'), component[2], type] for component in components]
    return filepaths_and_text


def load_voxpopuli(filename, type):
    with open(filename, encoding='utf-8') as f:
        lines = [line.strip().split('\t') for line in f][1:]  # First line is the header
        base = os.path.dirname(filename)
        filepaths_and_text = []
        for line in lines:
            if len(line) == 0:
                continue
            file, raw_text, norm_text, speaker_id, split, gender = line
            year = file[:4]
            filepaths_and_text.append([os.path.join(base, year, f'{file}.ogg.wav'), raw_text, type])
    return filepaths_and_text


class CharacterTokenizer:
    def encode(self, txt):
        return text_to_sequence(txt, ['english_cleaners'])

    def decode(self, seq):
        return sequence_to_text(seq)


class TextWavLoader(torch.utils.data.Dataset):
    def __init__(self, hparams):
        self.path = hparams['path']
        if not isinstance(self.path, list):
            self.path = [self.path]
        self.types = opt_get(hparams, ['types'], [0 for _ in self.path])

        fetcher_mode = opt_get(hparams, ['fetcher_mode'], 'lj')
        if not isinstance(fetcher_mode, list):
            fetcher_mode = [fetcher_mode]
        assert len(self.path) == len(fetcher_mode)

        self.load_conditioning = opt_get(hparams, ['load_conditioning'], False)
        self.conditioning_candidates = opt_get(hparams, ['num_conditioning_candidates'], 1)
        self.conditioning_length = opt_get(hparams, ['conditioning_length'], 44100)
        self.debug_failures = opt_get(hparams, ['debug_loading_failures'], False)
        self.load_aligned_codes = opt_get(hparams, ['load_aligned_codes'], False)
        self.aligned_codes_to_audio_ratio = opt_get(hparams, ['aligned_codes_ratio'], 443)
        self.audiopaths_and_text = []
        for p, fm, type in zip(self.path, fetcher_mode, self.types):
            if fm == 'lj' or fm == 'libritts':
                fetcher_fn = load_filepaths_and_text_type
            elif fm == 'tsv':
                fetcher_fn = load_tsv_aligned_codes_type if self.load_aligned_codes else load_tsv_type
            elif fm == 'mozilla_cv':
                assert not self.load_conditioning  # Conditioning inputs are incompatible with mozilla_cv
                fetcher_fn = load_mozilla_cv
            elif fm == 'voxpopuli':
                assert not self.load_conditioning  # Conditioning inputs are incompatible with voxpopuli
                fetcher_fn = load_voxpopuli
            else:
                raise NotImplementedError()
            self.audiopaths_and_text.extend(fetcher_fn(p, type))
        self.text_cleaners = hparams.text_cleaners
        self.sample_rate = hparams.sample_rate
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)
        self.max_wav_len = opt_get(hparams, ['max_wav_length'], None)
        if self.max_wav_len is not None:
            self.max_aligned_codes = self.max_wav_len // self.aligned_codes_to_audio_ratio
        self.max_text_len = opt_get(hparams, ['max_text_length'], None)
        assert self.max_wav_len is not None and self.max_text_len is not None
        self.use_bpe_tokenizer = opt_get(hparams, ['use_bpe_tokenizer'], True)
        if self.use_bpe_tokenizer:
            from data.audio.voice_tokenizer import VoiceBpeTokenizer
            self.tokenizer = VoiceBpeTokenizer(opt_get(hparams, ['tokenizer_vocab'], '../experiments/bpe_lowercase_asr_256.json'))
        else:
            self.tokenizer = CharacterTokenizer()
        self.skipped_items = 0  # records how many items are skipped when accessing an index.

    def get_wav_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text, type = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2]
        text_seq = self.get_text(text)
        wav = load_audio(audiopath, self.sample_rate)
        return (text_seq, wav, text, audiopath_and_text[0], type)

    def get_text(self, text):
        tokens = self.tokenizer.encode(text)
        tokens = torch.IntTensor(tokens)
        if self.use_bpe_tokenizer:
            # Assert if any UNK,start tokens encountered.
            assert not torch.any(tokens == 1)
        # The stop token should always be sacred.
        assert not torch.any(tokens == 0)
        return tokens

    def __getitem__(self, index):
        self.skipped_items += 1
        try:
            tseq, wav, text, path, type = self.get_wav_text_pair(self.audiopaths_and_text[index])
            if text is None or len(text.strip()) == 0:
                raise ValueError
            if wav is None or wav.shape[-1] < (.6 * self.sample_rate):
                # Ultra short clips are also useless (and can cause problems within some models).
                raise ValueError
            cond, cond_is_self = load_similar_clips(self.audiopaths_and_text[index][0], self.conditioning_length, self.sample_rate,
                                      n=self.conditioning_candidates) if self.load_conditioning else (None, False)
        except:
            if self.skipped_items > 100:
                raise  # Rethrow if we have nested too far.
            if self.debug_failures:
                print(f"error loading {self.audiopaths_and_text[index][0]} {sys.exc_info()}")
            return self[(index+1) % len(self)]

        if self.load_aligned_codes:
            aligned_codes = self.audiopaths_and_text[index][3]

        actually_skipped_items = self.skipped_items
        self.skipped_items = 0
        if wav is None or \
            (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len) or \
            (self.max_text_len is not None and tseq.shape[0] > self.max_text_len):
            # Basically, this audio file is nonexistent or too long to be supported by the dataset.
            # It's hard to handle this situation properly. Best bet is to return the a random valid token and skew the dataset somewhat as a result.
            if self.debug_failures:
                print(f"error loading {path}: ranges are out of bounds; {wav.shape[-1]}, {tseq.shape[0]}")
            rv = random.randint(0,len(self)-1)
            return self[rv]
        orig_output = wav.shape[-1]
        orig_text_len = tseq.shape[0]
        if wav.shape[-1] != self.max_wav_len:
            wav = F.pad(wav, (0, self.max_wav_len - wav.shape[-1]))
            if self.load_aligned_codes:
                # These codes are aligned to audio inputs, so make sure to pad them as well.
                aligned_codes = F.pad(aligned_codes, (0, self.max_aligned_codes-aligned_codes.shape[0]))
        if tseq.shape[0] != self.max_text_len:
            tseq = F.pad(tseq, (0, self.max_text_len - tseq.shape[0]))
        res = {
            'real_text': text,
            'padded_text': tseq,
            'text_lengths': torch.tensor(orig_text_len, dtype=torch.long),
            'wav': wav,
            'wav_lengths': torch.tensor(orig_output, dtype=torch.long),
            'filenames': path,
            'skipped_items': actually_skipped_items,
            'type': type,
        }
        if self.load_conditioning:
            res['conditioning'] = cond
            res['conditioning_contains_self'] = cond_is_self
        if self.load_aligned_codes:
            res['aligned_codes'] = aligned_codes
        return res

    def __len__(self):
        return len(self.audiopaths_and_text)


class PairedVoiceDebugger:
    def __init__(self):
        self.total_items = 0
        self.loaded_items = 0
        self.self_conditioning_items = 0

    def get_state(self):
        return {'total_items': self.total_items,
                'loaded_items': self.loaded_items,
                'self_conditioning_items': self.self_conditioning_items}

    def load_state(self, state):
        if isinstance(state, dict):
            self.total_items = opt_get(state, ['total_items'], 0)
            self.loaded_items = opt_get(state, ['loaded_items'], 0)
            self.self_conditioning_items = opt_get(state, ['self_conditioning_items'], 0)

    def update(self, batch):
        self.total_items += batch['wav'].shape[0]
        self.loaded_items += batch['skipped_items'].sum().item()
        if 'conditioning' in batch.keys():
            self.self_conditioning_items += batch['conditioning_contains_self'].sum().item()

    def get_debugging_map(self):
        return {
            'total_samples_loaded': self.total_items,
            'percent_skipped_samples': (self.loaded_items - self.total_items) / self.loaded_items,
            'percent_conditioning_is_self': self.self_conditioning_items / self.loaded_items,
        }


if __name__ == '__main__':
    batch_sz = 8
    params = {
        'mode': 'paired_voice_audio',
        'path': ['Y:\\libritts/test-clean_list.txt'],
        'fetcher_mode': ['libritts'],
        'phase': 'train',
        'n_workers': 0,
        'batch_size': batch_sz,
        'max_wav_length': 255995,
        'max_text_length': 200,
        'sample_rate': 22050,
        'load_conditioning': True,
        'num_conditioning_candidates': 2,
        'conditioning_length': 44000,
        'use_bpe_tokenizer': True,
        'load_aligned_codes': False,
    }
    from data import create_dataset, create_dataloader

    def save(b, i, ib, key, c=None):
        if c is not None:
            torchaudio.save(f'{i}_clip_{ib}_{key}_{c}.wav', b[key][ib][c], 22050)
        else:
            torchaudio.save(f'{i}_clip_{ib}_{key}.wav', b[key][ib], 22050)

    ds, c = create_dataset(params, return_collate=True)
    dl = create_dataloader(ds, params, collate_fn=c)
    i = 0
    m = None
    for i, b in tqdm(enumerate(dl)):
        for ib in range(batch_sz):
            print(f'{i} {ib} {b["real_text"][ib]}')
            save(b, i, ib, 'wav')
        if i > 5:
            break

