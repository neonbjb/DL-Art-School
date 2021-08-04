import os
import random
import numpy as np
import torch
import torch.utils.data
from torch import LongTensor
from tqdm import tqdm

import models.tacotron2.layers as layers
from models.tacotron2.taco_utils import load_wav_to_torch, load_filepaths_and_text

from models.tacotron2.text import text_to_sequence
from utils.util import opt_get
from models.tacotron2.text import symbols
import torch.nn.functional as F


class GptTtsDataset(torch.utils.data.Dataset):
    NUMBER_SYMBOLS = len(symbols)+3
    TEXT_START_TOKEN = LongTensor([NUMBER_SYMBOLS-3])
    TEXT_STOP_TOKEN = LongTensor([NUMBER_SYMBOLS-2])

    def __init__(self, opt):
        self.path = os.path.dirname(opt['path'])
        self.audiopaths_and_text = load_filepaths_and_text(opt['path'])
        self.text_cleaners=['english_cleaners']

        self.MEL_DICTIONARY_SIZE = opt['mel_vocab_size']+3
        self.MEL_START_TOKEN = LongTensor([self.MEL_DICTIONARY_SIZE-3])
        self.MEL_STOP_TOKEN = LongTensor([self.MEL_DICTIONARY_SIZE-2])

    def __getitem__(self, index):
        # Fetch text and add start/stop tokens.
        audiopath_and_text = self.audiopaths_and_text[index]
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        text = torch.cat([self.TEXT_START_TOKEN, text, self.TEXT_STOP_TOKEN], dim=0)

        # Fetch quantized MELs
        quant_path = audiopath.replace('wavs/', 'quantized_mels/') + '.pth'
        filename = os.path.join(self.path, quant_path)
        qmel = torch.load(filename)
        qmel = torch.cat([self.MEL_START_TOKEN, qmel, self.MEL_STOP_TOKEN])

        return text, qmel, audiopath

    def __len__(self):
        return len(self.audiopaths_and_text)


class GptTtsCollater():
    NUMBER_SYMBOLS = len(symbols)+3
    TEXT_PAD_TOKEN = NUMBER_SYMBOLS-1

    def __init__(self, opt):

        self.MEL_DICTIONARY_SIZE = opt['mel_vocab_size']+3
        self.MEL_PAD_TOKEN = self.MEL_DICTIONARY_SIZE-1

    def __call__(self, batch):
        text_lens = [len(x[0]) for x in batch]
        max_text_len = max(text_lens)
        mel_lens = [len(x[1]) for x in batch]
        max_mel_len = max(mel_lens)
        texts = []
        qmels = []
        for b in batch:
            text, qmel, _ = b
            texts.append(F.pad(text, (0, max_text_len-len(text)), value=self.TEXT_PAD_TOKEN))
            qmels.append(F.pad(qmel, (0, max_mel_len-len(qmel)), value=self.MEL_PAD_TOKEN))

        filenames = [j[2] for j in batch]

        return {
            'padded_text': torch.stack(texts),
            'input_lengths': LongTensor(text_lens),
            'padded_qmel': torch.stack(qmels),
            'output_lengths': LongTensor(mel_lens),
            'filenames': filenames
        }


if __name__ == '__main__':
    params = {
        'mode': 'gpt_tts',
        'path': 'E:\\audio\\LJSpeech-1.1\\ljs_audio_text_train_filelist.txt',
        'phase': 'train',
        'n_workers': 0,
        'batch_size': 16,
        'mel_vocab_size': 512,
    }
    from data import create_dataset, create_dataloader

    ds, c = create_dataset(params, return_collate=True)
    dl = create_dataloader(ds, params, collate_fn=c)
    i = 0
    m = []
    max_text = 0
    max_mel = 0
    for b in tqdm(dl):
        max_mel = max(max_mel, b['padded_qmel'].shape[2])
        max_text = max(max_text, b['padded_text'].shape[1])
    m=torch.stack(m)
    print(m.mean(), m.std())
