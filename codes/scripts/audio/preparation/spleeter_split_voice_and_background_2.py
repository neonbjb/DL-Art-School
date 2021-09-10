from scipy.io import wavfile
import os

import numpy as np
from scipy.io import wavfile
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.spleeter.separator import Separator
from scripts.audio.preparation.spleeter_dataset import SpleeterDataset


def main():
    src_dir = 'F:\\split\\podcast-dump0'
    output_dir = 'F:\\tmp\\out'
    output_dir_bg = 'F:\\tmp\\bg'
    output_sample_rate=22050
    batch_size=24

    dl = DataLoader(SpleeterDataset(src_dir, output_sample_rate), batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    separator = Separator('pretrained_models/2stems', input_sr=output_sample_rate)
    for e, batch in enumerate(tqdm(dl)):
        #if e < 406500:
        #    continue
        waves = batch['wave']
        paths = batch['path']
        durations = batch['duration']

        sep = separator.separate(waves)
        for j in range(sep['vocals'].shape[0]):
            vocals = sep['vocals'][j]
            bg = sep['accompaniment'][j]
            vmax = np.abs(vocals).mean()
            bmax = np.abs(bg).mean()

            # Only output to the "good" sample dir if the ratio of background noise to vocal noise is high enough.
            ratio = vmax / (bmax+.0000001)
            if ratio >= 25:  # These values were derived empirically
                od = output_dir
                out_sound = waves[j].cpu().numpy()
            elif ratio <= 1:
                od = output_dir_bg
                out_sound = bg
            else:
                continue

            # Strip out channels.
            if len(out_sound.shape) > 1:
                out_sound = out_sound[:, 0]  # Just use the first channel.
            # Resize to true duration
            out_sound = out_sound[:durations[j]]

            # Compile an output path.
            path = paths[j]
            reld = os.path.relpath(os.path.dirname(path), src_dir)
            os.makedirs(os.path.join(od, reld), exist_ok=True)
            relp = os.path.relpath(path, src_dir)
            output_path = os.path.join(od, relp)

            wavfile.write(output_path, output_sample_rate, out_sound)


# Uses torch spleeter to divide audio clips into one of two bins:
# 1. Audio has little to no background noise, saved to "output_dir"
# 2. Audio has a lot of background noise, bg noise split off and saved to "output_dir_bg"
if __name__ == '__main__':
    main()
