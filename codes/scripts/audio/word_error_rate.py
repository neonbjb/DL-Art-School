# Original source: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/deepspeech_pytorch/validation.py
import os

import Levenshtein as Lev
import torch
from tqdm import tqdm

from data.audio.voice_tokenizer import VoiceBpeTokenizer
from models.tacotron2.text import cleaners


def clean_text(text):
  for name in ['english_cleaners']:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


# Converts text to all-uppercase and separates punctuation from words.
def normalize_text(text):
    text = text.upper()
    for punc in ['.', ',', ':', ';']:
        text = text.replace(punc, f' {punc}')
    return text.strip()


class WordErrorRate:
    def calculate_metric(self, transcript, reference):
        wer_inst = self.wer_calc(transcript, reference)
        self.wer += wer_inst
        self.n_tokens += len(reference.split())

    def compute(self):
        wer = float(self.wer) / self.n_tokens
        return wer.item() * 100

    def wer_calc(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))


def load_truths(file):
    niltok = VoiceBpeTokenizer(None)
    out = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readline():
            spl = line.split('|')
            if len(spl) != 2:
                continue
            path, truth = spl
            path = path.replace('wav/', '')
            truth = niltok.preprocess_text(truth)  # This may or may not be considered a "cheat", but the model is only trained on preprocessed text.
            out[path] = truth
    return out


if __name__ == '__main__':
    inference_tsv = 'results.tsv'
    libri_base = '/h/bigasr_dataset/librispeech/test_clean/test_clean.txt'

    # Pre-process truth values
    truths = load_truths(libri_base)

    wer = WordErrorRate()
    wer_scores = []
    with open(inference_tsv, 'r') as tsv_file:
        tsv = tsv_file.read().splitlines()
        for line in tqdm(tsv):
            sentence_pred, wav = line.split('\t')
            sentence_pred = normalize_text(sentence_pred)
            sentence_real = normalize_text(truths[wav])
            wer_scores.append(wer.wer_calc(sentence_real, sentence_pred))
    print(f"WER: {torch.tensor(wer_scores, dtype=torch.float).mean()}")
