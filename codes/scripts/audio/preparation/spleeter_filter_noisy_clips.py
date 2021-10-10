import argparse

import numpy as np
from spleeter.separator import Separator
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.audio.preparation.spleeter_utils.spleeter_dataset import SpleeterDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--out')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--partition_size', default=None)
    parser.add_argument('--partition', default=None)
    args = parser.parse_args()

    src_dir = args.path
    out_file = args.out
    output_sample_rate=22050
    resume_file = args.resume

    loader = DataLoader(SpleeterDataset(src_dir, batch_sz=16, sample_rate=output_sample_rate,
                                        max_duration=10, partition=args.partition, partition_size=args.partition_size,
                                        resume=resume_file), batch_size=1, num_workers=1)

    separator = Separator('spleeter:2stems')
    unacceptable_files = open(out_file, 'a')
    for batch in tqdm(loader):
        audio, files, ends = batch['audio'], batch['files'], batch['ends']
        sep = separator.separate(audio.squeeze(0).numpy())
        vocals = sep['vocals']
        bg = sep['accompaniment']
        start = 0
        for path, end in zip(files, ends):
            vmax = np.abs(vocals[start:end]).mean()
            bmax = np.abs(bg[start:end]).mean()
            start = end

            # Only output to the "good" sample dir if the ratio of background noise to vocal noise is high enough.
            ratio = vmax / (bmax+.0000001)
            if ratio < 18:  # These values were derived empirically
                unacceptable_files.write(f'{path[0]}\n')
        unacceptable_files.flush()

    unacceptable_files.close()


if __name__ == '__main__':
    main()
