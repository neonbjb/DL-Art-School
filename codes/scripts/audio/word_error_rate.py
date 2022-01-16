import Levenshtein
from jiwer import wer, compute_measures
import torch
from tqdm import tqdm

from data.audio.voice_tokenizer import VoiceBpeTokenizer


def load_truths(file):
    niltok = VoiceBpeTokenizer(None)
    out = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            spl = line.split('|')
            if len(spl) != 2:
                print(spl)
                continue
            path, truth = spl
            #path = path.replace('wav/', '')
            # This preprocesses the truth data in the same way that training data is processed: removing punctuation, all lowercase, removing unnecessary
            # whitespace, and applying "english cleaners", which convert words like "mrs" to "missus" and such.
            truth = niltok.preprocess_text(truth)
            out[path] = truth
    return out


if __name__ == '__main__':
    inference_tsv = 'results.tsv'
    libri_base = 'y:\\bigasr_dataset/librispeech/test_clean/test_clean.txt'

    # Pre-process truth values
    truths = load_truths(libri_base)

    niltok = VoiceBpeTokenizer(None)
    ground_truths = []
    hypotheses = []
    with open(inference_tsv, 'r') as tsv_file:
        tsv = tsv_file.read().splitlines()
        for line in tqdm(tsv):
            sentence_pred, wav = line.split('\t')
            hypotheses.append(niltok.preprocess_text(sentence_pred))
            ground_truths.append(truths[wav])
    wer = wer(ground_truths, hypotheses)*100
    print(f"WER: {wer}")
