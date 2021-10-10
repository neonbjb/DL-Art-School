import multiprocessing
from math import ceil

from scipy.io import wavfile
import os

import argparse
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from spleeter.audio.adapter import AudioAdapter
from tqdm import tqdm

from data.util import IMG_EXTENSIONS
from scripts.audio.preparation.spleeter_separator_mod import Separator


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_wav_file(filename):
    return filename.endswith('.wav')


def is_audio_file(filename):
    AUDIO_EXTENSIONS = ['.wav', '.mp3', '.wma', 'm4b']
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def _get_paths_from_images(path, qualifier=is_image_file):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if qualifier(fname) and 'ref.jpg' not in fname:
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    if not images:
        print("Warning: {:s} has no valid image file".format(path))
    return images


def _get_paths_from_lmdb(dataroot):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes


def find_audio_files(dataroot, include_nonwav=False):
    if include_nonwav:
        return find_files_of_type(None, dataroot, qualifier=is_audio_file)[0]
    else:
        return find_files_of_type(None, dataroot, qualifier=is_wav_file)[0]


def find_files_of_type(data_type, dataroot, weights=[], qualifier=is_image_file):
    if isinstance(dataroot, list):
        paths = []
        for i in range(len(dataroot)):
            r = dataroot[i]
            extends = 1

            # Weights have the effect of repeatedly adding the paths from the given root to the final product.
            if weights:
                extends = weights[i]
            for j in range(extends):
                paths.extend(_get_paths_from_images(r, qualifier))
        paths = sorted(paths)
        sizes = len(paths)
    else:
        paths = sorted(_get_paths_from_images(dataroot, qualifier))
        sizes = len(paths)
    return paths, sizes


class SpleeterDataset(Dataset):
    def __init__(self, src_dir, batch_sz, max_duration, sample_rate=22050, partition=None, partition_size=None, resume=None):
        self.batch_sz = batch_sz
        self.max_duration = max_duration
        self.files = find_audio_files(src_dir, include_nonwav=True)
        self.sample_rate = sample_rate
        self.separator = Separator('spleeter:2stems', multiprocess=False, load_tf=False)

        # Partition files if needed.
        if partition_size is not None:
            psz = int(partition_size)
            prt = int(partition)
            self.files = self.files[prt * psz:(prt + 1) * psz]

        # Find the resume point and carry on from there.
        if resume is not None:
            for i, f in enumerate(self.files):
                if resume in f:
                    break
            assert i < len(self.files)
            self.files = self.files[i:]
        self.loader = AudioAdapter.default()

    def __len__(self):
        return ceil(len(self.files) / self.batch_sz)

    def __getitem__(self, item):
        item = item * self.batch_sz
        wavs = None
        files = []
        ends = []
        for k in range(self.batch_sz):
            ind = k+item
            if ind >= len(self.files):
                break

            try:
                wav, sr = self.loader.load(self.files[ind], sample_rate=self.sample_rate)
                assert sr == 22050
                # Get rid of all channels except one.
                if wav.shape[1] > 1:
                    wav = wav[:, 0]

                if wavs is None:
                    wavs = wav
                else:
                    wavs = np.concatenate([wavs, wav])
                ends.append(wavs.shape[0])
                files.append(self.files[ind])
            except:
                print(f'Error loading {self.files[ind]}')
        stft = self.separator.stft(wavs)
        return {
            'audio': wavs,
            'files': files,
            'ends': ends,
            'stft': stft
        }

def invert_spectrogram_and_save(args, queue):
    separator = Separator('spleeter:2stems', multiprocess=False, load_tf=False)
    out_file = args.out
    unacceptable_files = open(out_file, 'a')

    while True:
        combo = queue.get()
        if combo is None:
            break
        vocals, bg, wavlen, files, ends = combo
        vocals = separator.stft(vocals, inverse=True, length=wavlen)
        bg = separator.stft(vocals, inverse=True, length=wavlen)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--out')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--partition_size', default=None)
    parser.add_argument('--partition', default=None)
    args = parser.parse_args()

    src_dir = args.path
    output_sample_rate=22050
    resume_file = args.resume

    worker_queue = multiprocessing.Queue()
    from scripts.audio.preparation.useless import invert_spectrogram_and_save
    worker = multiprocessing.Process(target=invert_spectrogram_and_save, args=(args, worker_queue))
    worker.start()

    loader = DataLoader(SpleeterDataset(src_dir, batch_sz=16, sample_rate=output_sample_rate,
                                        max_duration=10, partition=args.partition, partition_size=args.partition_size,
                                        resume=resume_file), batch_size=1, num_workers=0)

    separator = Separator('spleeter:2stems', multiprocess=False)
    for k in range(100):
        for batch in tqdm(loader):
            audio, files, ends, stft = batch['audio'], batch['files'], batch['ends'], batch['stft']
            sep = separator.separate_spectrogram(stft.squeeze(0).numpy())
            worker_queue.put((sep['vocals'], sep['accompaniment'], audio.shape[1], files, ends))
    worker_queue.put(None)
    worker.join()


if __name__ == '__main__':
    main()
