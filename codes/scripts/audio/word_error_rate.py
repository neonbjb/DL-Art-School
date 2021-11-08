# Original source: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/deepspeech_pytorch/validation.py
import os

import Levenshtein as Lev
import torch
from tqdm import tqdm

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


if __name__ == '__main__':
    inference_tsv = 'D:\\dlas\\codes\\46000ema_8beam.tsv'
    libri_base = 'Z:\\libritts\\test-clean'

    wer = WordErrorRate()
    wer_scores = []
    with open(inference_tsv, 'r') as tsv_file:
        tsv = tsv_file.read().splitlines()
        for line in tqdm(tsv):
            sentence_pred, wav = line.split('\t')
            sentence_pred = normalize_text(sentence_pred)

            wav_comp = wav.split('_')
            reader = wav_comp[0]
            book = wav_comp[1]
            txt_file = os.path.join(libri_base, reader, book, wav.replace('.wav', '.normalized.txt'))
            with open(txt_file, 'r') as txt_file_hndl:
                txt_uncleaned = txt_file_hndl.read()
                sentence_real = normalize_text(clean_text(txt_uncleaned))
                wer_scores.append(wer.wer_calc(sentence_real, sentence_pred))
    print(f"WER: {torch.tensor(wer_scores, dtype=torch.float).mean()}")