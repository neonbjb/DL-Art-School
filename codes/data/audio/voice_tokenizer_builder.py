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
    trainer = BpeTrainer(special_tokens=['[STOP]', '[UNK]'], vocab_size=9999)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(['all_texts.txt'], trainer)
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