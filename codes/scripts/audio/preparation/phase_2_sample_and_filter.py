import argparse
import functools
import os
import shutil
import sys
from multiprocessing.pool import ThreadPool
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainer.injectors.audio_injectors import MelSpectrogramInjector
from utils.util import find_audio_files, load_audio, load_model_from_config, ceil_multiple


class AudioFolderDataset(torch.utils.data.Dataset):
    def __init__(self, path, sampling_rate, pad_to, skip=0):
        self.audiopaths = find_audio_files(path)[skip:]
        self.sampling_rate = sampling_rate
        self.pad_to = pad_to

    def __getitem__(self, index):
        try:
            path = self.audiopaths[index]
            audio_norm = load_audio(path, self.sampling_rate)
        except:
            print(f"Error loading audio for file {path} {sys.exc_info()}")
            # Recover gracefully. It really sucks when we outright fail.
            return self[index+1]

        orig_length = audio_norm.shape[-1]
        if audio_norm.shape[-1] > self.pad_to:
            print(f"Warning - {path} has a longer audio clip than is allowed: {audio_norm.shape[-1]}; allowed: {self.pad_to}. "
                  f"Truncating the clip, though this will likely invalidate the prediction.")
            audio_norm = audio_norm[:self.pad_to]
        else:
            padding = self.pad_to - audio_norm.shape[-1]
            if padding > 0:
                audio_norm = torch.nn.functional.pad(audio_norm, (0, padding))

        return {
            'clip': audio_norm,
            'samples': orig_length,
            'path': path
        }

    def __len__(self):
        return len(self.audiopaths)


def process_folder(folder, output_path, base_path, progress_file, max_files):
    classifier = load_model_from_config(args.classifier_model_opt, model_name='classifier', also_load_savepoint=True).cuda().eval()
    dataset = AudioFolderDataset(folder, sampling_rate=22050, pad_to=600000)
    if len(dataset) == 0:
        return
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    spec_injector = MelSpectrogramInjector({'in': 'clip', 'out': 'mel'}, {})

    with torch.no_grad():
        total_count = 0
        for batch in tqdm(dataloader):
            try:
                max_len = max(batch['samples'])
                clips = batch['clip'][:, :max_len].cuda()
                paths = batch['path']
                mels = spec_injector({'clip': clips})['mel']

                def get_spec_mags(clip):
                    stft = torch.stft(clip, n_fft=22000, hop_length=1024, return_complex=True)
                    stft = stft[0, -2000:, :]
                    return (stft.real ** 2 + stft.imag ** 2).sqrt()
                no_hifreq_data = get_spec_mags(clips).mean(dim=1) < .01
                if torch.all(no_hifreq_data):
                    continue

                labels = torch.argmax(classifier(mels), dim=-1)

                for b in range(clips.shape[0]):
                    if no_hifreq_data[b]:
                        continue
                    if labels[b] != 0:
                        continue
                    dirpath = paths[b].replace(os.path.basename(paths[b]), "")
                    path = os.path.relpath(dirpath, base_path)
                    opath = os.path.join(output_path, path)
                    os.makedirs(opath, exist_ok=True)
                    shutil.copy(paths[b], opath)
                    total_count += 1
                    if total_count >= max_files:
                        break
                if total_count >= max_files:
                    break
            except:
                print("Exception encountered. Will ignore and continue. Exception info follows.")
                print(sys.exc_info())

    with open(progress_file, 'a', encoding='utf-8') as pf:
        pf.write(folder + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to search for split files (should be the direct output of phase 1)',
                        default='Y:\\clips\\red_rising_split')
    parser.add_argument('--progress_file', type=str, help='Place to store all folders that have already been processed', default='Y:\\clips\\red_rising_filtered\\already_processed.txt')
    parser.add_argument('--output_path', type=str, help='Path where sampled&filtered files are sent', default='Y:\\clips\\red_rising_filtered')
    parser.add_argument('--num_threads', type=int, help='Number of concurrent workers processing files.', default=6)
    parser.add_argument('--max_samples_per_folder', type=int, help='Maximum number of clips that can be extracted from each folder.', default=999999)
    parser.add_argument('--classifier_model_opt', type=str, help='Train/test options file that configures the model used to classify the audio clips.',
                        default='../options/test_noisy_audio_clips_classifier.yml')
    args = parser.parse_args()

    # Build a list of split audio files to process
    all_split_files = []
    for cast_dir in os.listdir(args.path):
        fullpath = os.path.join(args.path, cast_dir)
        if os.path.isdir(fullpath):
            all_split_files.append(fullpath)
    shuffle(all_split_files)
    all_split_files = set(all_split_files)

    # Load the already processed files, if present, and get the set difference.
    if os.path.exists(args.progress_file):
        with open(args.progress_file, 'r', encoding='utf-8') as pf:
            processed = set([l.strip() for l in pf.readlines()])
        orig_len = len(all_split_files)
        all_split_files = all_split_files - processed
        print(f'All folders: {orig_len}, processed files: {len(processed)}; {len(all_split_files)/orig_len}% of files remain to be processed.')

    with ThreadPool(args.num_threads) as pool:
        list(tqdm(pool.imap(functools.partial(process_folder, output_path=args.output_path, base_path=args.path,
                                              progress_file=args.progress_file, max_files=args.max_samples_per_folder), all_split_files), total=len(all_split_files)))
