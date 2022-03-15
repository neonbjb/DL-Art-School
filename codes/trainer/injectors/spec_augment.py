# Original source: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/deepspeech_pytorch/loader/sparse_image_warp.py
# Removes the time_warp augmentation and only implements masking.

import numpy as np
import random

import torch
import torchvision.utils

from trainer.inject import Injector
from utils.util import opt_get


def spec_augment(mel_spectrogram, frequency_masking_para=27, time_masking_para=5, frequency_mask_num=1, time_mask_num=1):

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


def test_mel_mask():
    from data.audio.unsupervised_audio_dataset import load_audio
    from trainer.injectors.base_injectors import MelSpectrogramInjector
    spec_maker = MelSpectrogramInjector({'in': 'audio', 'out': 'spec'}, {})
    a = load_audio('D:\\data\\audio\\libritts\\test-clean\\61\\70970\\61_70970_000007_000001.wav', 22050).unsqueeze(0)
    s = spec_maker({'audio': a})['spec']
    visualization_spectrogram(s, 'original spec')
    saug = spec_augment(s, 50, 5, 1, 3)
    visualization_spectrogram(saug, 'modified spec')


'''
Crafty bespoke injector that is used when training ASR models to create longer sequences to ensure that the entire
input length embedding is trained. Does this by concatenating every other batch element together to create longer
sequences which (theoretically) use similar amounts of GPU memory.
'''
class CombineMelInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.audio_key = opt['audio_key']
        self.text_key = opt['text_key']
        self.audio_lengths = opt['audio_lengths_key']
        self.text_lengths = opt['text_lengths_key']
        self.output_audio_key = opt['output_audio_key']
        self.output_text_key = opt['output_text_key']
        from models.audio.tts.tacotron2 import symbols
        self.text_separator = len(symbols)+1  # Probably need to allow this to be set by user.

    def forward(self, state):
        audio = state[self.audio_key]
        texts = state[self.text_key]
        audio_lengths = state[self.audio_lengths]
        text_lengths = state[self.text_lengths]
        if audio.shape[0] == 1:
            return {self.output_audio_key: audio, self.output_text_key: texts}
        combined_audios = []
        combined_texts = []
        for b in range(audio.shape[0]//2):
            a1 = audio[b*2, :, :audio_lengths[b*2]]
            a2 = audio[b*2+1, :, :audio_lengths[b*2+1]]
            a = torch.cat([a1, a2], dim=1)
            a = torch.nn.functional.pad(a, (0, audio.shape[-1]*2-a.shape[-1]))
            combined_audios.append(a)

            t1 = texts[b*2, :text_lengths[b*2]]
            t1 = torch.nn.functional.pad(t1, (0, 1), value=self.text_separator)
            t2 = texts[b*2+1, :text_lengths[b*2+1]]
            t = torch.cat([t1, t2], dim=0)
            t = torch.nn.functional.pad(t, (0, texts.shape[-1]*2-t.shape[-1]))
            combined_texts.append(t)
        return {self.output_audio_key: torch.stack(combined_audios, dim=0),
                self.output_text_key: torch.stack(combined_texts, dim=0)}


def test_mel_injector():
    inj = CombineMelInjector({'audio_key': 'a', 'text_key': 't', 'audio_lengths_key': "alk", 'text_lengths_key': 'tlk'}, {})
    a = torch.rand((4, 22000))
    al = torch.tensor([11000,14000,22000,20000])
    t = torch.randint(0, 120, (4, 250))
    tl = torch.tensor([100,120,200,250])
    rs = inj({'a': a, 't': t, 'alk': al, 'tlk': tl})



if __name__ == '__main__':
    test_mel_injector()