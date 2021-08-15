from scipy.io import wavfile
from spleeter.separator import Separator
from tqdm import tqdm

from data.util import find_audio_files
import os.path as osp
from spleeter.audio.adapter import AudioAdapter
import numpy as np


if __name__ == '__main__':
    src_dir = 'P:\\Audiobooks-Podcasts'
    #src_dir = 'E:\\audio\\books'
    output_dir = 'D:\\data\\audio\\misc-split'
    output_dir_lq = 'D:\\data\\audio\\misc-split-with-bg'
    output_dir_garbage = 'D:\\data\\audio\\misc-split-garbage'
    #output_dir = 'E:\\audio\\books-clips'
    clip_length = 5  # In seconds
    sparsity = .1  # Only this proportion of the total clips are extracted as wavs.
    output_sample_rate=22050

    audio_loader = AudioAdapter.default()
    separator = Separator('spleeter:2stems')
    files = find_audio_files(src_dir, include_nonwav=True)
    for e, file in enumerate(tqdm(files)):
        if e < 1092:
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
            wave, sample_rate = audio_loader.load(file, sample_rate=output_sample_rate)
        except:
            print(f"Error with {file}")
            continue

        #if len(wave.shape) < 2:
        #    continue

        # Calculate how much data we need to extract for each clip.
        clip_sz = sample_rate * clip_length
        interval = int(sample_rate * (clip_length / sparsity))
        i = 0
        while (i+clip_sz) < wave.shape[0]:
            clip = wave[i:i+clip_sz]
            sep = separator.separate(clip)
            vocals = sep['vocals']
            bg = sep['accompaniment']
            vmax = np.abs(vocals).mean()
            bmax = np.abs(bg).mean()

            # Only output to the "good" sample dir if the ratio of background noise to vocal noise is high enough.
            ratio = vmax / (bmax+.0000001)
            if ratio >= 25:  # These values were derived empirically
                od = output_dir
                os = clip
            elif ratio >= 1:
                od = output_dir_lq
                os = vocals
            else:
                od = output_dir_garbage
                os = vocals

            # Strip out channels.
            if len(os.shape) > 1:
                os = os[:, 0]  # Just use the first channel.

            wavfile.write(osp.join(od, f'{e}_{file_basis}_{i}.wav'), output_sample_rate, os)
            i = i + interval
