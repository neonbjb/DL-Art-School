from scipy.io import wavfile
from spleeter.separator import Separator
from tqdm import tqdm
'''
Uses a model configuration to load a classifier and iterate through a dataset, binning each class into it's own
folder.
'''

from data.util import find_audio_files
import os
import os.path as osp
from spleeter.audio.adapter import AudioAdapter
import numpy as np


# Uses spleeter_utils to divide audio clips into one of two bins:
# 1. Audio has little to no background noise, saved to "output_dir"
# 2. Audio has a lot of background noise, bg noise split off and saved to "output_dir_bg"
if __name__ == '__main__':
    src_dir = 'F:\\split\\joe_rogan'
    output_dir = 'F:\\split\\cleaned\\joe_rogan'
    output_dir_bg = 'F:\\split\\background-noise\\joe_rogan'
    output_sample_rate=22050

    os.makedirs(output_dir_bg, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    audio_loader = AudioAdapter.default()
    separator = Separator('spleeter:2stems')
    files = find_audio_files(src_dir, include_nonwav=True)
    for e, file in enumerate(tqdm(files)):
        #if e < 406500:
        #    continue
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

        sep = separator.separate(wave)
        vocals = sep['vocals']
        bg = sep['accompaniment']
        vmax = np.abs(vocals).mean()
        bmax = np.abs(bg).mean()

        # Only output to the "good" sample dir if the ratio of background noise to vocal noise is high enough.
        ratio = vmax / (bmax+.0000001)
        if ratio >= 25:  # These values were derived empirically
            od = output_dir
            os = wave
        elif ratio <= 1:
            od = output_dir_bg
            os = bg
        else:
            continue

        # Strip out channels.
        if len(os.shape) > 1:
            os = os[:, 0]  # Just use the first channel.

        wavfile.write(osp.join(od, file_basis, f'{e}.wav'), output_sample_rate, os)
