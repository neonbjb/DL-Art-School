

"""
Master script that processes all MP3 files found in an input directory. Splits those files up into sub-files of a
predetermined duration.
"""
import argparse
import functools
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from trainer.injectors.audio_injectors import MusicCheaterLatentInjector, normalize_torch_mel, denormalize_mel

from models.audio.music.transformer_diffusion14 import get_cheater_encoder_v2


def report_progress(progress_file, file):
    with open(progress_file, 'a', encoding='utf-8') as f:
        f.write(f'{file}\n')


def process_folder(file, model, base_path, output_path, progress_file):
    outdir = os.path.join(output_path, f'{os.path.relpath(os.path.dirname(file), base_path)}')
    os.makedirs(outdir, exist_ok=True)
    with np.load(file) as npz_file:
        mel = torch.tensor(npz_file['arr_0']).cuda().unsqueeze(0)
        # Fix the normalization issues with the old mels. This should get reverted when these mels are re-generated.
        mel = normalize_torch_mel(denormalize_mel(mel))
        assert mel.min() > -1.001 and mel.max() < 1.001
        model = model.cuda()
        with torch.no_grad():
            cheater = model(mel)
        np.savez(os.path.join(outdir, os.path.basename(file)), cheater.cpu().numpy())
    report_progress(progress_file, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to search for files', default='Y:\\separated\\large_mels')
    parser.add_argument('--progress_file', type=str, help='Place to store all files that have already been processed', default='Y:\\separated\\large_mel_cheaters\\already_processed.txt')
    parser.add_argument('--output_path', type=str, help='Path for output files', default='Y:\\separated\\large_mel_cheaters')
    parser.add_argument('--num_threads', type=int, help='Number of concurrent workers processing files (there must be a GPU per-worker.)', default=1)
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
        root_music_files = set(path.rglob("*.npz"))
        torch.save(root_music_files, cache_path)

    orig_len = len(root_music_files)
    folders = list(root_music_files - processed_files)
    print(f"Found {len(folders)} files to process. Total processing is {100 * (orig_len - len(folders)) / orig_len}% complete.")

    k = 0
    for k in range(args.num_threads-1):
        if os.fork() == 0:
            break

    # k is now the process number.
    partition_len = (len(folders)//args.num_threads)+1
    folders = folders[k*partition_len:(k+1)*partition_len]

    model = get_cheater_encoder_v2().eval().cpu()
    model.load_state_dict(torch.load('../experiments/tfd14_cheater_encoder.pth', map_location=torch.device('cpu')))
    model = model.to(f'cuda:{k}')

    for folder in tqdm(folders):
        process_folder(folder, model=model, output_path=args.output_path, base_path=args.path, progress_file=args.progress_file)
