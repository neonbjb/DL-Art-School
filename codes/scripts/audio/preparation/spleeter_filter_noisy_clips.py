from scipy.io import wavfile
import os

import argparse
import numpy as np
from scipy.io import wavfile
from spleeter.separator import Separator
from tqdm import tqdm
from spleeter.audio.adapter import AudioAdapter
from tqdm import tqdm


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
    waiting_for_file = args.resume is not None
    resume_file = args.resume

    audio_loader = AudioAdapter.default()
    files = find_audio_files(src_dir, include_nonwav=True)

    # Partition files if needed.
    if args.partition_size is not None:
        psz = int(args.partition_size)
        prt = int(args.partition)
        files = files[prt*psz:(prt+1)*psz]
    
    #separator = Separator('pretrained_models/2stems', input_sr=output_sample_rate)
    separator = Separator('spleeter:2stems')
    unacceptable_files = open(out_file, 'a')
    for e, path in enumerate(tqdm(files)):
        if waiting_for_file and resume_file not in path:
            continue
        waiting_for_file = False
        print(f"{e}: Processing {path}")
        spleeter_ld, sr = audio_loader.load(path, sample_rate=output_sample_rate)
        sep = separator.separate(spleeter_ld)
        vocals = sep['vocals']
        bg = sep['accompaniment']
        vmax = np.abs(vocals).mean()
        bmax = np.abs(bg).mean()
        
        # Only output to the "good" sample dir if the ratio of background noise to vocal noise is high enough.
        ratio = vmax / (bmax+.0000001)
        if ratio < 25:  # These values were derived empirically
            unacceptable_files.write(f'{path}\n')
        unacceptable_files.flush()

    unacceptable_files.close()


if __name__ == '__main__':
    main()
