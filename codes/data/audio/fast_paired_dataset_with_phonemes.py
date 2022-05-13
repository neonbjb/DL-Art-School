import hashlib
import os
import random
import sys
import time
from itertools import groupby

import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Processor

from data.audio.paired_voice_audio_dataset import CharacterTokenizer
from data.audio.unsupervised_audio_dataset import load_audio, load_similar_clips
from utils.util import opt_get


def parse_tsv_aligned_codes(line, base_path):
    fpt = line.strip().split('\t')
    def convert_string_list_to_tensor(strlist):
        if strlist.startswith('['):
            strlist = strlist[1:]
        if strlist.endswith(']'):
            strlist = strlist[:-1]
        as_ints = [int(s) for s in strlist.split(', ')]
        return torch.tensor(as_ints)
    return os.path.join(base_path, f'{fpt[1]}'), fpt[0], convert_string_list_to_tensor(fpt[2])


class FastPairedVoiceDataset(torch.utils.data.Dataset):
    """
    This dataset is derived from paired_voice_audio, but it only supports loading from TSV files generated from the
    ocotillo transcription engine, which includes alignment codes. To support the vastly larger TSV files, this dataset
    uses an indexing mechanism which randomly selects offsets within the translation file to seek to. The data returned
    is relative to these offsets.

    In practice, this means two things:
    1) Index {i} of this dataset means nothing: fetching from the same index will almost always return different data.
       As a result, this dataset should not be used for validation or test runs. Use PairedVoiceAudio dataset instead.
    2) This dataset has a slight bias for items with longer text or longer filenames.

    The upshot is that this dataset loads extremely quickly and consumes almost no system memory.
    """
    def __init__(self, hparams):
        self.paths = hparams['path']
        phoneme_paths = hparams['phoneme_paths']
        self.paths = [(p, False) for p in self.paths] + [(p, True) for p in phoneme_paths]

        self.paths_size_bytes = [os.path.getsize(p) for p, _ in self.paths]
        self.total_size_bytes = sum(self.paths_size_bytes)
        self.types = opt_get(hparams, ['types'], [0 for _ in self.paths])

        self.normal_text_end_token = hparams['normal_text_end_token']
        self.load_conditioning = opt_get(hparams, ['load_conditioning'], False)
        self.conditioning_candidates = opt_get(hparams, ['num_conditioning_candidates'], 1)
        self.conditioning_length = opt_get(hparams, ['conditioning_length'], 44100)
        self.produce_ctc_metadata = opt_get(hparams, ['produce_ctc_metadata'], False)
        self.debug_failures = opt_get(hparams, ['debug_loading_failures'], False)
        self.text_cleaners = hparams.text_cleaners
        self.sample_rate = hparams.sample_rate
        self.aligned_codes_to_audio_ratio = 443 * self.sample_rate // 22050
        self.max_wav_len = opt_get(hparams, ['max_wav_length'], None)
        self.load_aligned_codes = opt_get(hparams, ['load_aligned_codes'], False)
        if self.max_wav_len is not None:
            self.max_aligned_codes = self.max_wav_len // self.aligned_codes_to_audio_ratio
        self.max_text_len = opt_get(hparams, ['max_text_length'], None)
        assert self.max_wav_len is not None and self.max_text_len is not None
        self.use_bpe_tokenizer = opt_get(hparams, ['use_bpe_tokenizer'], False)
        if self.use_bpe_tokenizer:
            from data.audio.voice_tokenizer import VoiceBpeTokenizer
            self.tokenizer = VoiceBpeTokenizer(opt_get(hparams, ['tokenizer_vocab'], '../experiments/bpe_lowercase_asr_256.json'))
        else:
            self.tokenizer = CharacterTokenizer()
        self.ipa_phoneme_tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft").tokenizer
        self.ipa_phoneme_tokenizer.do_phonemize = False
        self.skipped_items = 0  # records how many items are skipped when accessing an index.

        self.load_times = torch.zeros((256,))
        self.load_ind = 0

    def get_wav_text_pair(self, audiopath_and_text, is_phonetic):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text_seq = self.get_text(text, is_phonetic)
        wav = load_audio(audiopath, self.sample_rate)
        return (text_seq, wav, text, audiopath_and_text[0])

    def get_text(self, text, is_phonetic):
        if is_phonetic:
            tokens = self.ipa_phoneme_tokenizer.encode(text)
        else:
            tokens = self.tokenizer.encode(text)
        tokens = torch.IntTensor(tokens)
        if self.use_bpe_tokenizer:
            # Assert if any UNK,start tokens encountered.
            assert not torch.any(tokens == 1)
        # The stop token should always be sacred.
        assert not torch.any(tokens == 0)
        return tokens

    def load_random_line(self, depth=0):
        assert depth < 10

        rand_offset = random.randint(0, self.total_size_bytes)
        for i in range(len(self.paths)):
            if rand_offset < self.paths_size_bytes[i]:
                break
            else:
                rand_offset -= self.paths_size_bytes[i]
        path, is_phonetic = self.paths[i]
        type = self.types[i]
        with open(path, 'r', encoding='utf-8') as f:
            f.seek(rand_offset)
            # Read the rest of the line we seeked to, then the line after that.
            try:  # This can fail when seeking to a UTF-8 escape byte.
                f.readline()
            except:
                return self.load_random_line(depth=depth + 1)  # On failure, just recurse and try again.
            l2 = f.readline()

        if l2:
            try:
                base_path = os.path.dirname(path)
                return parse_tsv_aligned_codes(l2, base_path), type, is_phonetic
            except:
                print(f"error parsing random offset: {sys.exc_info()}")
        return self.load_random_line(depth=depth+1)  # On failure, just recurse and try again.

    def get_ctc_metadata(self, codes):
        grouped = groupby(codes.tolist())
        rcodes, repeats, seps = [], [], [0]
        for val, group in grouped:
            if val == 0:
                seps[-1] = len(list(group))  # This is a very important distinction! It means the padding belongs to the character proceeding it.
            else:
                rcodes.append(val)
                repeats.append(len(list(group)))
                seps.append(0)

        rcodes = torch.tensor(rcodes)
        # These clip values are sane maximum values which I did not see in the datasets I have access to.
        repeats = torch.clip(torch.tensor(repeats), min=1, max=30)
        seps = torch.clip(torch.tensor(seps[:-1]), max=120)

        # Pad or clip the codes to get them to exactly self.max_text_len
        orig_lens = rcodes.shape[0]
        if rcodes.shape[0] < self.max_text_len:
            gap = self.max_text_len - rcodes.shape[0]
            rcodes = F.pad(rcodes, (0, gap))
            repeats = F.pad(repeats, (0, gap), value=1)  # The minimum value for repeats is 1, hence this is the pad value too.
            seps = F.pad(seps, (0, gap))
        elif rcodes.shape[0] > self.max_text_len:
            rcodes = rcodes[:self.max_text_len]
            repeats = rcodes[:self.max_text_len]
            seps = seps[:self.max_text_len]
        return {
            'ctc_raw_codes': rcodes,
            'ctc_separators': seps,
            'ctc_repeats': repeats,
            'ctc_raw_lengths': orig_lens,
        }

    def __getitem__(self, index):
        start = time.time()
        self.skipped_items += 1
        apt, type, is_phonetic = self.load_random_line()
        try:
            tseq, wav, text, path = self.get_wav_text_pair(apt, is_phonetic)
            if text is None or len(text.strip()) == 0:
                raise ValueError
            cond, cond_is_self = load_similar_clips(apt[0], self.conditioning_length, self.sample_rate,
                                      n=self.conditioning_candidates) if self.load_conditioning else (None, False)
        except:
            if self.skipped_items > 100:
                raise  # Rethrow if we have nested too far.
            if self.debug_failures:
                print(f"error loading {apt[0]} {sys.exc_info()}")
            return self[(index+1) % len(self)]
        raw_codes = apt[2]
        aligned_codes = raw_codes

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

        # Shift phonetic token and aligned_code tokens over.
        if is_phonetic:
            tseq = tseq + self.normal_text_end_token
            # But keep the padding/stop tokens.
            if self.load_aligned_codes:
                aligned_codes = aligned_codes + self.normal_text_end_token

        orig_output = wav.shape[-1]
        orig_text_len = tseq.shape[0]
        orig_aligned_code_length = aligned_codes.shape[0]
        if wav.shape[-1] != self.max_wav_len:
            wav = F.pad(wav, (0, self.max_wav_len - wav.shape[-1]))
            # These codes are aligned to audio inputs, so make sure to pad them as well.
            aligned_codes = F.pad(aligned_codes, (0, self.max_aligned_codes-aligned_codes.shape[0]))
        if tseq.shape[0] != self.max_text_len:
            tseq = F.pad(tseq, (0, self.max_text_len - tseq.shape[0]))

        elapsed = time.time() - start
        self.load_times[self.load_ind] = elapsed
        self.load_ind = (self.load_ind + 1) % len(self.load_times)

        res = {
            'real_text': text,
            'padded_text': tseq,
            'text_lengths': torch.tensor(orig_text_len, dtype=torch.long),
            'wav': wav,
            'wav_lengths': torch.tensor(orig_output, dtype=torch.long),
            'filenames': path,
            'skipped_items': actually_skipped_items,
            'load_time': self.load_times.mean(),
            'type': type,
        }
        if self.load_conditioning:
            res['conditioning'] = cond
            res['conditioning_contains_self'] = cond_is_self
        if self.load_aligned_codes:
            res['aligned_codes']: aligned_codes
            res['aligned_codes_lengths']: orig_aligned_code_length
        if self.produce_ctc_metadata:
            res.update(self.get_ctc_metadata(raw_codes))

        return res

    def __len__(self):
        return self.total_size_bytes // 1000  # 1000 cuts down a TSV file to the actual length pretty well.


class FastPairedVoiceDebugger:
    def __init__(self):
        self.total_items = 0
        self.loaded_items = 0
        self.self_conditioning_items = 0
        self.unique_files = set()
        self.load_time = 0

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
        self.load_time = batch['load_time'].mean().item()
        for filename in batch['filenames']:
            self.unique_files.add(hashlib.sha256(filename.encode('utf-8')))
        if 'conditioning' in batch.keys():
            self.self_conditioning_items += batch['conditioning_contains_self'].sum().item()

    def get_debugging_map(self):
        return {
            'total_samples_loaded': self.total_items,
            'percent_skipped_samples': (self.loaded_items - self.total_items) / self.loaded_items,
            'percent_conditioning_is_self': self.self_conditioning_items / self.loaded_items,
            'unique_files_loaded': len(self.unique_files),
            'load_time': self.load_time,
        }


if __name__ == '__main__':
    batch_sz = 16
    params = {
        'mode': 'fast_paired_voice_audio_with_phonemes',
        'path': ['y:/libritts/train-clean-100/transcribed-oco.tsv',],
        'phoneme_paths': ['y:/libritts/train-other-500/transcribed-phoneme-oco.tsv'],
        'types': [0,0],
        'normal_text_end_token': 256,
        'phase': 'train',
        'n_workers': 0,
        'batch_size': batch_sz,
        'max_wav_length': 220500,
        'max_text_length': 500,
        'sample_rate': 22050,
        'load_conditioning': True,
        'num_conditioning_candidates': 2,
        'conditioning_length': 102400,
        'use_bpe_tokenizer': True,
        'load_aligned_codes': False,
        'debug_loading_failures': True,
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
    max_pads, max_repeats = 0, 0
    for i, b in tqdm(enumerate(dl)):
        for ib in range(batch_sz):
            #max_pads = max(max_pads, b['ctc_pads'].max())
            #max_repeats = max(max_repeats, b['ctc_repeats'].max())
            print(f'{i} {ib} {b["real_text"][ib]}')
            #save(b, i, ib, 'wav')
            #save(b, i, ib, 'conditioning', 0)
            #save(b, i, ib, 'conditioning', 1)
            pass
        if i > 15:
            break
    print(max_pads, max_repeats)

