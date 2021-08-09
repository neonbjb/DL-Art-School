import audio2numpy
from scipy.io import wavfile
from tqdm import tqdm

from data.util import find_audio_files
import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp

if __name__ == '__main__':
    src_dir = 'O:\\podcast_dumps'
    #src_dir = 'E:\\audio\\books'
    output_dir = 'D:\\data\\audio\\podcasts-split'
    #output_dir = 'E:\\audio\\books-clips'
    clip_length = 5  # In seconds
    sparsity = .05  # Only this proportion of the total clips are extracted as wavs.
    output_sample_rate=22050

    files = find_audio_files(src_dir, include_nonwav=True)
    for e, file in enumerate(tqdm(files)):
        if e < 1486:
            continue
        file_basis = osp.relpath(file, src_dir)\
            .replace('/', '_')\
            .replace('\\', '_')\
            .replace('.', '_')\
            .replace(' ', '_')\
            .replace('!', '_')\
            .replace(',', '_')
        if len(file_basis) > 100:
            file_basis = file_basis[:100]
        try:
            wave, sample_rate = audio2numpy.open_audio(file)
        except:
            print(f"Error with {file}")
            continue
        wave = torch.tensor(wave)
        # Strip out channels.
        if len(wave.shape) > 1:
            wave = wave[:, 1]  # Just use the first channel.

        # Calculate how much data we need to extract for each clip.
        clip_sz = sample_rate * clip_length
        interval = int(sample_rate * (clip_length / sparsity))
        i = 0
        if wave.shape[-1] == 0:
            print("Something went wrong: wave shape is 0.")
        while (i+clip_sz) < wave.shape[-1]:
            clip = wave[i:i+clip_sz]
            clip = F.interpolate(clip.view(1,1,clip_sz), scale_factor=output_sample_rate/sample_rate).squeeze()
            wavfile.write(osp.join(output_dir, f'{e}_{file_basis}_{i}.wav'), output_sample_rate, clip.numpy())
            i = i + interval
