import argparse
import os

import numpy
import torch
from spleeter.audio.adapter import AudioAdapter
from tqdm import tqdm

from data.util import find_audio_files
# Uses pydub to process a directory of audio files, splitting them into clips at points where it detects a small amount
# of silence.
from trainer.injectors.base_injectors import MelSpectrogramInjector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()
    files = find_audio_files(args.path, include_nonwav=True)
    mel_inj = MelSpectrogramInjector({'in':'in', 'out':'out'}, {})
    audio_loader = AudioAdapter.default()
    for e, wav_file in enumerate(tqdm(files)):
        if e < 0:
            continue
        print(f"Processing {wav_file}..")
        outfile = f'{wav_file}.npz'
        if os.path.exists(outfile):
            continue

        try:
            wave, sample_rate = audio_loader.load(wav_file, sample_rate=22050)
            wave = torch.tensor(wave)[:,0].unsqueeze(0)
            wave = wave / wave.abs().max()
        except:
            print(f"Error with {wav_file}")
            continue

        inj = mel_inj({'in': wave})
        numpy.savez_compressed(outfile, inj['out'].numpy())


if __name__ == '__main__':
    main()
