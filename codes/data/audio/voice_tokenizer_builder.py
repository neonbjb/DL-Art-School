import re

import datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from data.audio.paired_voice_audio_dataset import load_mozilla_cv, load_voxpopuli, load_tsv
from models.tacotron2.taco_utils import load_filepaths_and_text


def build_text_file_from_priors(priors, output):
    with open(output, 'w', encoding='utf-8') as out:
        for p, fm in priors:
            if fm == 'lj' or fm == 'libritts':
                fetcher_fn = load_filepaths_and_text
            elif fm == 'tsv':
                fetcher_fn = load_tsv
            elif fm == 'mozilla_cv':
                fetcher_fn = load_mozilla_cv
            elif fm == 'voxpopuli':
                fetcher_fn = load_voxpopuli
            else:
                raise NotImplementedError()
            apt = fetcher_fn(p)
            for path, text in apt:
                out.write(text + "\n")
            out.flush()


def train():
    with open('all_texts.txt', 'r', encoding='utf-8') as at:
        ttsd = at.readlines()
    bcd = datasets.load_dataset('bookcorpus', cache_dir='Z:\\huggingface_datasets\\cache')['train']
    wkd = datasets.load_dataset('wikipedia', '20200501.en', cache_dir='Z:\\huggingface_datasets\\cache')['train']

    allowed_characters_re = re.compile(r'^[a-z!@#%_=:;"/, \-\$\^&\*\(\)\+\{\[\]\}\\\.]+$')
    def preprocess_word(word):
        word = word.lower()
        if not bool(allowed_characters_re.match(word)):
            return ''
        return word

    def batch_iterator(batch_size=1000):
        print("Processing ASR texts.")
        for i in range(0, len(ttsd), batch_size):
            yield [preprocess_word(t) for t in ttsd[i:i+batch_size]]

        print("Processing bookcorpus.")
        for i in range(0, len(bcd), batch_size):
            yield [preprocess_word(t) for t in bcd[i:i+batch_size]['text']]

        print("Processing wikipedia.")
        for i in range(0, len(wkd), batch_size):
            yield [preprocess_word(t) for t in wkd[i:i+batch_size]['text']]

    trainer = BpeTrainer(special_tokens=['[STOP]', '[UNK]'], vocab_size=9999)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(ttsd)+len(bcd)+len(wkd))
    tokenizer.save('gpt_tts_tokenizer.json')


if __name__ == '__main__':
    '''
    build_text_file_from_priors([('Y:\\bigasr_dataset\\libritts\\train-all.txt', 'libritts'),
                                 ('Y:\\bigasr_dataset\\libritts\\test-clean_list.txt', 'libritts'),
                                 #('Y:\\bigasr_dataset\\voxpopuli\\audio\\transcribed_data\\en\\asr_en.tsv', 'voxpopuli'),
                                 ('Y:\\bigasr_dataset\\voxpopuli\\audio\\transcribed_data\\en\\asr_train.tsv', 'voxpopuli'),
                                 ('Y:\\clips\\books1-transcribed.tsv', 'tsv'),
                                 ('Y:\\clips\\books2-transcribed.tsv', 'tsv'),
                                 ('Y:\\clips\\podcasts-0-transcribed.tsv', 'tsv')], 'all_texts.txt')
    '''
    train()