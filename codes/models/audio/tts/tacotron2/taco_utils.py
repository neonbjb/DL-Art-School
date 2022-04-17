import os.path

import numpy as np
import torch
from scipy.io.wavfile import read


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len)).to(lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    if data.dtype == np.int32:
        norm_fix = 2 ** 31
    elif data.dtype == np.int16:
        norm_fix = 2 ** 15
    elif data.dtype == np.float16 or data.dtype == np.float32:
        norm_fix = 1.
    else:
        raise NotImplemented(f"Provided data dtype not supported: {data.dtype}")
    return (torch.FloatTensor(data.astype(np.float32)) / norm_fix, sampling_rate)


def load_filepaths_and_text_type(filename, type, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [list(line.strip().split(split)) + [type] for line in f]
        base = os.path.dirname(filename)
        for j in range(len(filepaths_and_text)):
            filepaths_and_text[j][0] = os.path.join(base, filepaths_and_text[j][0])
    return filepaths_and_text

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
        base = os.path.dirname(filename)
        for j in range(len(filepaths_and_text)):
            filepaths_and_text[j][0] = os.path.join(base, filepaths_and_text[j][0])
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)