

"""
Master script that processes all MP3 files found in an input directory. Splits those files up into sub-files of a
predetermined duration.
"""
import argparse
import functools
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

from data.util import find_audio_files, find_files_of_type
from trainer.injectors.audio_injectors import TorchMelSpectrogramInjector
from utils.util import load_audio


def report_progress(progress_file, file):
    with open(progress_file, 'a', encoding='utf-8') as f:
        f.write(f'{file}\n')


spec_fn = TorchMelSpectrogramInjector({'n_mel_channels': 256, 'mel_fmax': 11000, 'filter_length': 16000,
                                       'true_normalization': True, 'normalize': True, 'in': 'in', 'out': 'out'}, {}).cuda()

def produce_mel(audio):
    return spec_fn({'in': audio.unsqueeze(0)})['out'].squeeze(0)


def process_folder(folder, base_path, output_path, progress_file, max_duration, sampling_rate=22050):
    outdir = os.path.join(output_path, f'{os.path.relpath(folder, base_path)}')
    os.makedirs(outdir, exist_ok=True)

    files = list(os.listdir(folder))
    i = 0
    output_i = 0
    while i < len(files):
        last_ordinal = -1
        total_progress = 0
        to_combine = []
        while i < len(files) and total_progress < max_duration:
            audio_file = os.path.join(folder, files[i], "no_vocals.wav")
            if not os.path.exists(audio_file):
                break
            to_combine.append(load_audio(audio_file, 22050))
            file_ordinal = int(files[i])
            if last_ordinal != -1 and file_ordinal != last_ordinal+1:
                last_ordinal = file_ordinal
                continue
            else:
                i += 1
                total_progress += 30
        if total_progress > 30:
            combined = torch.cat(to_combine, dim=-1).cuda()
            mel = produce_mel(combined)
            assert mel.max() < 1.00001, mel.max()
            assert mel.min() > -1.00001, mel.min()
            mel = mel.cpu().numpy()
            np.savez(os.path.join(outdir, f'{output_i}'), mel)
            output_i += 1
    report_progress(progress_file, folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to search for files', default='Y:\\separated')
    parser.add_argument('--progress_file', type=str, help='Place to store all files that have already been processed', default='Y:\\separated\\large_mels\\already_processed.txt')
    parser.add_argument('--output_path', type=str, help='Path for output files', default='Y:\\separated\\large_mels')
    parser.add_argument('--num_threads', type=int, help='Number of concurrent workers processing files.', default=3)
    parser.add_argument('--max_duration', type=int, help='Duration per clip in seconds', default=120)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    processed_files = set()
    if os.path.exists(args.progress_file):
        with open(args.progress_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                processed_files.add(line.strip())

    cache_path = os.path.join(args.output_path, 'cache.pth')
    if os.path.exists(cache_path):
        root_music_files = torch.load(cache_path)
    else:
        path = Path(args.path)
        def collect(p):
            return str(os.path.dirname(os.path.dirname(p)))
        root_music_files = {collect(p) for p in path.rglob("*no_vocals.wav")}
        torch.save(root_music_files, cache_path)

    orig_len = len(root_music_files)
    folders = root_music_files - processed_files
    print(f"Found {len(folders)} files to process. Total processing is {100 * (orig_len - len(folders)) / orig_len}% complete.")

    with ThreadPool(args.num_threads) as pool:
        list(tqdm(pool.imap(functools.partial(process_folder, output_path=args.output_path, base_path=args.path,
                                              progress_file=args.progress_file, max_duration=args.max_duration), folders), total=len(folders)))
