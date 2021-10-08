from typing import Optional

import torch
import torch.nn as nn
from scipy.signal.windows import hann
from spleeter.audio.adapter import AudioAdapter
from torch.utils.data import Dataset
import numpy as np
import librosa

from data.util import find_audio_files


def spleeter_stft(
        data: np.ndarray, inverse: bool = False, length: Optional[int] = None
) -> np.ndarray:
    """
    Single entrypoint for both stft and istft. This computes stft and
    istft with librosa on stereo data. The two channels are processed
    separately and are concatenated together in the result. The
    expected input formats are: (n_samples, 2) for stft and (T, F, 2)
    for istft.

    Parameters:
        data (numpy.array):
            Array with either the waveform or the complex spectrogram
            depending on the parameter inverse
        inverse (bool):
            (Optional) Should a stft or an istft be computed.
        length (Optional[int]):

    Returns:
        numpy.ndarray:
            Stereo data as numpy array for the transform. The channels
            are stored in the last dimension.
    """
    assert not (inverse and length is None)
    data = np.asfortranarray(data)
    N = 4096
    H = 1024
    win = hann(N, sym=False)
    fstft = librosa.core.istft if inverse else librosa.core.stft
    win_len_arg = {"win_length": None, "length": None} if inverse else {"n_fft": N}
    n_channels = data.shape[-1]
    out = []
    for c in range(n_channels):
        d = (
            np.concatenate((np.zeros((N,)), data[:, c], np.zeros((N,))))
            if not inverse
            else data[:, :, c].T
        )
        s = fstft(d, hop_length=H, window=win, center=False, **win_len_arg)
        if inverse:
            s = s[N: N + length]
        s = np.expand_dims(s.T, 2 - inverse)
        out.append(s)
    if len(out) == 1:
        return out[0]
    return np.concatenate(out, axis=2 - inverse)


class SpleeterDataset(Dataset):
    def __init__(self, src_dir, sample_rate=22050, max_duration=20, skip=0):
        self.files = find_audio_files(src_dir, include_nonwav=True)
        if skip > 0:
            self.files = self.files[skip:]
        self.audio_loader = AudioAdapter.default()
        self.sample_rate = sample_rate
        self.max_duration = max_duration

    def __getitem__(self, item):
        file = self.files[item]
        wave, sample_rate = self.audio_loader.load(file, sample_rate=self.sample_rate)
        assert sample_rate == self.sample_rate
        stft = torch.tensor(spleeter_stft(wave))
        # TODO: pad this up so it can be batched.
        return {
            'path': file,
            'wave': wave,
            'stft': stft,
            #'duration': original_duration,
        }

    def __len__(self):
        return len(self.files)