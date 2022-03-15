import os

import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from munch import munchify
from tqdm import tqdm

from data.audio.unsupervised_audio_dataset import UnsupervisedAudioDataset
from data.text.hf_datasets_wrapper import HfDataset
from utils.util import opt_get


def build_paired_voice_dataset(args):
    from data.audio.paired_voice_audio_dataset import TextWavLoader as D
    from models.audio.tts.tacotron2 import create_hparams
    default_params = create_hparams()
    default_params.update(args)
    dataset_opt = munchify(default_params)
    return D(dataset_opt)


class GrandConjoinedDataset(torch.utils.data.Dataset):
    """
    A joint text & speech dataset that joins three separate datasets into a single batch:
    1. Unpaired text
    2. Unpaired speech
    3. Paired speech & text

    Supports situations where the underlying data sources for these three elements are differently sized, e.g. you can
    have a massive text corpus of 1B elements, a smaller unpaired speech corpus, and a small paired speech<->text corpus.

    Performs tokenization at this level, ignoring any tokenization performed by upstream datasets.
    """
    def __init__(self, opt):
        sample_rate = 22050  # Fixed.
        paired_dataset_args = opt['paired_dataset_args']
        self.only_paired = opt_get(opt, ['only_paired'], False)
        if not self.only_paired:
            unsupervised_audio_args = opt['unsupervised_audio_args']
            text_corpus_args = opt['text_corpus_args']

        self.max_paired_audio_length = opt['max_paired_audio_length']
        self.max_paired_text_length = opt['max_paired_text_length']
        self.max_solo_audio_length = opt['max_solo_audio_length']
        self.max_solo_text_length = opt['max_solo_text_length']
        self.collate = opt_get(opt, ['needs_collate'], False)
        self.sample_rate = sample_rate
        self.num_conditioning_candidates = opt_get(opt, ['num_conditioning_candidates'], 0)
        self.conditioning_length = opt_get(opt, ['conditioning_length'], 44000)
        load_conditioning = self.num_conditioning_candidates > 0

        # Set some sane arguments for all three datasets.
        paired_dataset_args['needs_collate'] = self.collate
        paired_dataset_args['load_conditioning'] = load_conditioning
        paired_dataset_args['num_conditioning_candidates'] = self.num_conditioning_candidates
        paired_dataset_args['conditioning_length'] = self.conditioning_length
        paired_dataset_args['sample_rate'] = sample_rate
        paired_dataset_args['max_wav_length'] = self.max_paired_audio_length
        paired_dataset_args['max_text_length'] = self.max_paired_text_length
        self.speech_and_text = build_paired_voice_dataset(paired_dataset_args)

        if not self.only_paired:
            unsupervised_audio_args['sampling_rate'] = sample_rate
            unsupervised_audio_args['do_augmentation'] = False
            unsupervised_audio_args['resample_clip'] = False
            unsupervised_audio_args['extra_samples'] = self.num_conditioning_candidates
            unsupervised_audio_args['extra_sample_length'] = self.conditioning_length
            if not self.collate:
                unsupervised_audio_args['pad_to_samples'] = self.max_solo_audio_length
            self.speech = UnsupervisedAudioDataset(unsupervised_audio_args)
            self.text = HfDataset(**text_corpus_args)

    def fetch_text_at(self, i):
        try:
            txt = self.text[i % len(self.text)]['text']
            assert '*' not in txt  # This is a hack to get around the use of '*' to mask expletives in some text-only datasets. There really isn't a linguistic use for this character anyways.
            tok = self.speech_and_text.get_text(txt)
            padding_required = self.max_solo_text_length - tok.shape[0]
            if padding_required < 0:
                # Just truncate since there is no conditioning required.
                tok = tok[:self.max_solo_text_length]
            elif padding_required > 0:
                tok = F.pad(tok, (0, padding_required))
            return txt, tok
        except:
            # This is fully expected: there are a lot of text strings we intentionally do not
            # handle (e.g. ones with emojis, or other languages). Just return another one.
            return self.fetch_text_at((i+1) % len(self.text))

    def fetch_snt_at(self, i):
        fetched = self.speech_and_text[i % len(self.speech_and_text)]
        if self.collate:
            tseq, wav, path, text, cond = fetched
            res = {
                'real_text': text,
                'padded_text': tseq,
                'text_lengths': torch.tensor(tseq.shape[0], dtype=torch.long),
                'wav': wav,
                'wav_lengths': torch.tensor(wav.shape[-1], dtype=torch.long),
                'filenames': path
            }
            if self.num_conditioning_candidates > 0:
                res['conditioning'] = cond
            return res
        else:
            return fetched

    def optionally_add_conditioning_candidates(self, res, paired, solo=None):
        if self.num_conditioning_candidates > 0:
            if solo is None:
                res['paired_audio_conditioning'] = paired['conditioning']
                res['speech_audio_conditioning'] = paired['conditioning']
            else:
                res['paired_audio_conditioning'] = paired['conditioning']
                res['speech_audio_conditioning'] = solo['alt_clips']
        return res

    def __getitem__(self, i):
        snt = self.fetch_snt_at(i)
        if self.only_paired:
            return self.optionally_add_conditioning_candidates({
                'paired_audio': snt['wav'],
                'paired_audio_lengths': snt['wav_lengths'],
                'paired_text': snt['real_text'],
                'paired_text_tokens': snt['padded_text'],
                'paired_file': snt['filenames'],
                'speech_audio': snt['wav'],
                'speech_audio_lengths': snt['wav_lengths'],
                'speech_file': snt['filenames'],
                'text_text': snt['real_text'],
                'text_tokens': snt['padded_text'],
            }, snt)
        else:
            txt, txt_tok = self.fetch_text_at(i % len(self.text))
            sp = self.speech[i % len(self.speech)]
            # Set upper bound on solo speech lengths. This is handled automatically when collation is turned off, but needs to be done otherwise.
            sp['clip'] = sp['clip'][:, :self.max_solo_audio_length]
            sp['clip_lengths'] = sp['clip_lengths'].clamp(0, self.max_solo_audio_length)
            return self.optionally_add_conditioning_candidates({
                'paired_audio': snt['wav'],
                'paired_audio_lengths': snt['wav_lengths'],
                'paired_text': snt['real_text'],
                'paired_text_tokens': snt['padded_text'],
                'paired_file': snt['filenames'],
                'speech_audio': sp['clip'],
                'speech_audio_lengths': sp['clip_lengths'],
                'speech_file': sp['path'],
                'text_text': txt,
                'text_tokens': txt_tok,
            }, snt, sp)

    def __len__(self):
        if self.only_paired:
            return len(self.speech_and_text)
        else:
            return max(len(self.speech), len(self.speech_and_text), len(self.text))


if __name__ == '__main__':
    batch_sz = 8
    train_params = {
        'mode': 'grand_conjoined_voice',
        'phase': 'train',
        'n_workers': 0,
        'batch_size': batch_sz,

        'max_paired_audio_length': 255995,
        'max_paired_text_length': 100,
        'max_solo_text_length': 200,
        'max_solo_audio_length': 307195,
        'num_conditioning_candidates': 1,
        'conditioning_length': 44000,
        'needs_collate': True,
        'paired_dataset_args': {
            'path': ['Z:\\bigasr_dataset\\tedlium\\train-all.txt'],
            'fetcher_mode': ['libritts'],
            'use_bpe_tokenizer': False,
        },
        'unsupervised_audio_args': {
            'path': ['Y:\\clips\\podcasts-0\\6175_20170425-How the National Security Council Works'],
            'cache_path': 'test_cache_delete_me2.pth',
        },
        'text_corpus_args': {
            'corpi': [['bookcorpus', '']],
            'cache_path': 'Z:\\huggingface_datasets\\cache',
        },
    }
    val_params = {
        'mode': 'grand_conjoined_voice',
        'phase': 'val',
        'n_workers': 0,
        'batch_size': batch_sz,

        'max_paired_audio_length': 255995,
        'max_paired_text_length': 200,
        'max_solo_text_length': 330,
        'max_solo_audio_length': 300000,
        'only_paired': True,
        'needs_collate': False,
        'paired_dataset_args': {
            'path': ['Z:\\bigasr_dataset\\libritts\\test-clean_list.txt'],
            'fetcher_mode': ['libritts'],
            'use_bpe_tokenizer': False,
        },
    }
    from data import create_dataset, create_dataloader
    os.remove('test_cache_delete_me2.pth')

    ds, c = create_dataset(train_params, return_collate=True)
    dl = create_dataloader(ds, train_params, collate_fn=c)

    def save(b, i, ib, key, c=None):
        if c is not None:
            torchaudio.save(f'{i}_clip_{ib}_{key}_{c}.wav', b[key][ib][c], 22050)
        else:
            torchaudio.save(f'{i}_clip_{ib}_{key}.wav', b[key][ib], 22050)

    def decode(b, ib, key):
        return ds.speech_and_text.tokenizer.decode(b[key][ib].cpu().numpy())

    i = 0
    m = None
    for i, b in tqdm(enumerate(dl)):
        for ib in range(batch_sz):
            #save(b, i, ib, 'paired_audio')
            #save(b, i, ib, 'paired_audio_conditioning', 0)
            #save(b, i, ib, 'paired_audio_conditioning', 1)
            print(f'Paired file: {b["paired_file"][ib]} text: {b["paired_text"][ib]}')
            print(f'Paired text decoded: {decode(b, ib, "paired_text_tokens")}')
            #save(b, i, ib, 'speech_audio')
            #save(b, i, ib, 'speech_audio_conditioning', 0)
            #save(b, i, ib, 'speech_audio_conditioning', 1)
            #print(f'Text: {b["text_text"][ib]}')
            #print(f'Text decoded: {decode(b, ib, "text_tokens")}')
        if i > 5:
            break

