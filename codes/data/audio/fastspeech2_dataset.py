import torch
from munch import munchify
from torch.utils.data import Dataset, DataLoader

import re
import numpy as np
import math
import os
import random
from models.fastspeech2.utils import pad_1D, pad_2D
from models.fastspeech2.symbols import symbols
import cleaners


""" from https://github.com/keithito/tacotron """
# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_to_sequence(text, cleaner_names):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      The text can optionally have ARPAbet sequences enclosed in curly braces embedded
      in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through

      Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)

        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(
            _clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'


class Dataset(Dataset):
    def __init__(self, opt):
        filename = "train.txt"
        sort = True
        hp = munchify(opt)
        self.basename, self.text = self.process_meta(os.path.join(hp.preprocessed_path, filename))
        self.sort = sort

        if hp.dataset == "VCTK":
            from data import vctk
            self.spk_table, self.inv_spk_table = vctk.get_spk_table()
        if hp.dataset == "LibriTTS":
            from data import libritts
            self.spk_table, self.inv_spk_table = libritts.get_spk_table()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        phone = np.array(text_to_sequence(self.text[idx], []))

        mel_path = os.path.join(hp.preprocessed_path, "mel", "{}-mel-{}.npy".format(hp.dataset, basename))
        mel_target = np.load(mel_path)

        D_path = os.path.join(hp.preprocessed_path, "alignment", "{}-ali-{}.npy".format(hp.dataset, basename))
        D = np.load(D_path)

        f0_path = os.path.join(hp.preprocessed_path, "f0", "{}-f0-{}.npy".format(hp.dataset, basename))
        f0 = np.load(f0_path)

        energy_path = os.path.join(hp.preprocessed_path, "energy", "{}-energy-{}.npy".format(hp.dataset, basename))
        energy = np.load(energy_path)

        sample = {"id": basename,
                  "text": phone,
                  "mel_target": mel_target,
                  "D": D,
                  "f0": f0,
                  "energy": energy}

        return sample

    def collate_fn(self, batch):
        len_arr = np.array([d["text"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = int(math.sqrt(batchsize))

        cut_list = list()
        for i in range(real_batchsize):
            if self.sort:
                cut_list.append(index_arr[i * real_batchsize:(i + 1) * real_batchsize])
            else:
                cut_list.append(np.arange(i * real_batchsize, (i + 1) * real_batchsize))

        output = list()
        for i in range(real_batchsize):
            output.append(self.reprocess(batch, cut_list[i]))

        # shuffle batch of batchs to solve the problem that
        # during synth, it always synthesizes short(long) sentences
        # 2020.10.03 KaiWei

        random.shuffle(output)
        return output

    def reprocess(self, batch, cut_list):
        ids = [batch[ind]["id"] for ind in cut_list]
        if hp.use_spk_embed:
            if hp.dataset == "VCTK" or hp.dataset == "LibriTTS":
                spk_ids = [self.spk_table[_id.split("_")[0]] for _id in ids]
            else:
                raise NotImplementedError("Looking up dataset {} speaker table not implemented".format(hp.dataset))

        texts = [batch[ind]["text"] for ind in cut_list]
        mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
        Ds = [batch[ind]["D"] for ind in cut_list]
        f0s = [batch[ind]["f0"] for ind in cut_list]
        energies = [batch[ind]["energy"] for ind in cut_list]

        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print(text, text.shape, D, D.shape, id_)

        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])

        texts = pad_1D(texts)
        Ds = pad_1D(Ds)
        mel_targets = pad_2D(mel_targets)
        f0s = pad_1D(f0s)
        energies = pad_1D(energies)
        log_Ds = np.log(Ds + hp.log_offset)

        out = {"id": ids,
               "text": texts,
               "mel_target": mel_targets,
               "D": Ds,
               "log_D": log_Ds,
               "f0": f0s,
               "energy": energies,
               "src_len": length_text,
               "mel_len": length_mel}
        if hp.use_spk_embed:
            out.update({"spk_ids": spk_ids})

        return out

    def process_meta(self, meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            text = []
            name = []
            for line in f.readlines():
                # 100_122655_000027_000001|{HH IY1 R IH0 Z UW1 M D}
                n, t = line.strip('\n').split('|')
                name.append(n)
                text.append(t)
            return name, text


if __name__ == "__main__":
    # Test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset('val.txt')
    training_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn,
                                 drop_last=True, num_workers=0)
    total_step = len(training_loader) * 8

    cnt = 0
    for i, batchs in enumerate(training_loader):
        for j, data_of_batch in enumerate(batchs):
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            if mel_target.shape[1] == D.sum().item():
                cnt += 1

    print(cnt, len(dataset))