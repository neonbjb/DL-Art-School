# Original source: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/deepspeech_pytorch/loader/sparse_image_warp.py
# Removes the time_warp augmentation and only implements masking.

import numpy as np
import random
import torchvision.utils

from trainer.inject import Injector
from utils.util import opt_get


def spec_augment(mel_spectrogram, frequency_masking_para=27, time_masking_para=70, frequency_mask_num=1, time_mask_num=1):

    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        if v - f < 0:
            continue
        f0 = random.randint(0, v-f)
        mel_spectrogram[:, f0:f0+f, :] = 0

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        if tau - t < 0:
            continue
        t0 = random.randint(0, tau-t)
        mel_spectrogram[:, :, t0:t0+t] = 0

    return mel_spectrogram

class MelMaskInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.freq_mask_sz = opt_get(opt, ['frequency_mask_size_high'], 27)
        self.n_freq_masks = opt_get(opt, ['frequency_mask_count'], 1)
        self.time_mask_sz = opt_get(opt, ['time_mask_size_high'], 5)
        self.n_time_masks = opt_get(opt, ['time_mask_count'], 3)

    def forward(self, state):
        h = state[self.input]
        return {self.output: spec_augment(h, self.freq_mask_sz, self.time_mask_sz, self.n_freq_masks, self.n_time_masks)}

def visualization_spectrogram(spec, title):
    # Turns spec into an image and outputs it to the filesystem.
    spec = spec.unsqueeze(dim=1)
    # Normalize so spectrogram is easier to view.
    spec = (spec - spec.mean()) / spec.std()
    spec = ((spec + 1) / 2).clip(0, 1)
    torchvision.utils.save_image(spec, f'{title}.png')

if __name__ == '__main__':
    from data.audio.unsupervised_audio_dataset import load_audio
    from trainer.injectors.base_injectors import MelSpectrogramInjector
    spec_maker = MelSpectrogramInjector({'in': 'audio', 'out': 'spec'}, {})
    a = load_audio('D:\\data\\audio\\libritts\\test-clean\\61\\70970\\61_70970_000007_000001.wav', 22050).unsqueeze(0)
    s = spec_maker({'audio': a})['spec']
    visualization_spectrogram(s, 'original spec')
    saug = spec_augment(s, 50, 5, 1, 3)
    visualization_spectrogram(saug, 'modified spec')
