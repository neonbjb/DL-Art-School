

"""
Master script that processes all MP3 files found in an input directory. Splits those files up into sub-files of a
predetermined duration.
"""
import argparse
import functools
import os
from multiprocessing.pool import ThreadPool

import torch
import torchaudio
from tqdm import tqdm

from data.util import find_audio_files, find_files_of_type
from utils.util import load_audio


def report_progress(progress_file, file):
    with open(progress_file, 'a', encoding='utf-8') as f:
        f.write(f'{file}\n')


def process_file(file, base_path, output_path, progress_file, duration_per_clip, sampling_rate=22050):
    lp = os.path.basename(file).lower()
    if ' live' in lp or 'concert' in lp:
        print(f"Skipping file {file} because likely a live performance")
        report_progress(progress_file, file)
        return
    try:
        audio = load_audio(file, sampling_rate)
    except:
        print(f"Error loading file {file}")
        report_progress(progress_file, file)
        return

    outdir = os.path.join(output_path, f'{os.path.relpath(file, base_path)[:-4]}').replace('.', '').strip()
    os.makedirs(outdir, exist_ok=True)
    splits = torch.split(audio, duration_per_clip * sampling_rate, dim=-1)
    for i, spl in enumerate(splits):
        if spl.shape[-1] != duration_per_clip*sampling_rate:
            continue  # In general, this just means "skip the last item".
        torchaudio.save(f'{outdir}/{i:05d}.wav', spl.unsqueeze(0), sampling_rate, encoding="PCM_S")
    report_progress(progress_file, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, help='Path to search for files', default='Y:\\sources\\soundcloud-mixes\\mixes2')
    parser.add_argument('-progress_file', type=str, help='Place to store all files that have already been processed', default='Y:\\sources\\soundcloud-mixes\\mixes2\\already_processed.txt')
    parser.add_argument('-output_path', type=str, help='Path for output files', default='Y:\\split\\soundcloud-mixes2')
    parser.add_argument('-num_threads', type=int, help='Number of concurrent workers processing files.', default=4)
    parser.add_argument('-duration', type=int, help='Duration per clip in seconds', default=30)
    args = parser.parse_args()

    processed_files = set()
    if os.path.exists(args.progress_file):
        with open(args.progress_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                processed_files.add(line.strip())

    files = set(find_audio_files(args.path, include_nonwav=True))
    orig_len = len(files)
    files = files - processed_files
    print(f"Found {len(files)} files to process. Total processing is {100*(orig_len-len(files))/orig_len}% complete.")

    with ThreadPool(args.num_threads) as pool:
        list(tqdm(pool.imap(functools.partial(process_file, output_path=args.output_path, base_path=args.path,
                                              progress_file=args.progress_file, duration_per_clip=args.duration), files), total=len(files)))
