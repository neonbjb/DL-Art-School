import random

import torch
import torchaudio.sox_effects

from models.audio.tts.tacotron2.taco_utils import load_wav_to_torch


# Returns random double on [l,h] as a string
def rdstr(l=0,h=1):
    assert h > l
    i=h-l
    return str(random.random() * i + l)


# Returns a randint on [s,e] as a string
def rdi(e, s=0):
    return str(random.randint(s,e))


class WavAugmentor:
    def __init__(self):
        pass

    def augment(self, wav, sample_rate):
        speed_effect = ['speed', rdstr(.8, 1)]
        '''
        Band effects are disabled until I can audit them better.
        band_effects = [
            ['reverb', '-w'],
            ['reverb'],
            ['band', rdi(8000, 3000), rdi(1000, 100)],
            ['bandpass', rdi(8000, 3000), rdi(1000, 100)],
            ['bass', rdi(20,-20)],
            ['treble', rdi(20,-20)],
            ['dither'],
            ['equalizer', rdi(3000, 100), rdi(1000, 100), rdi(10, -10)],
            ['hilbert'],
            ['sinc', '3k'],
            ['sinc', '-4k'],
            ['sinc', '3k-4k']
        ]
        band_effect = random.choice(band_effects)
        '''
        volume_effects = [
            ['loudness', rdi(10,-2)],
            ['overdrive', rdi(20,0), rdi(20,0)],
        ]
        vol_effect = random.choice(volume_effects)
        effects = [speed_effect, vol_effect]
        out, sr = torchaudio.sox_effects.apply_effects_tensor(wav, sample_rate, effects)
        # Add a variable amount of noise
        out = out + torch.rand_like(out) * random.random() * .03
        return out


if __name__ == '__main__':
    sample, _ = load_wav_to_torch('obama1.wav')
    sample = sample / 32768.0
    aug = WavAugmentor()
    for j in range(10):
        out = aug.augment(sample, 24000)
        torchaudio.save(f'out{j}.wav', out, 24000)