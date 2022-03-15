import os

import torch

from data.util import find_files_of_type, is_audio_file
from trainer.injectors.audio_injectors import MelSpectrogramInjector
from utils.util import load_audio

if __name__ == '__main__':
    path = 'C:\\Users\\jbetk\\Documents\\tmp\\some_audio'

    inj = MelSpectrogramInjector({'in': 'wav', 'out': 'mel',
                                  'mel_fmax': 12000, 'sampling_rate': 22050, 'n_mel_channels': 100
                                  },{})
    audio = find_files_of_type('img', path, qualifier=is_audio_file)[0]
    for clip in audio:
        if not clip.endswith('.wav'):
            continue
        wav = load_audio(clip, 24000)
        mel = inj({'wav': wav.unsqueeze(0)})['mel']
        torch.save(mel, clip.replace('.wav', '.mel'))