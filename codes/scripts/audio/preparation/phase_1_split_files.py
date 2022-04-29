

"""
Master script that processes all MP3 files found in an input directory. Performs the following operations, per-file:
1. Splits the file on silence intervals, throwing out all clips that are too short or long.
2.
"""
import argparse
import functools
import os
from multiprocessing.pool import ThreadPool

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from pydub.silence import split_on_silence
from tqdm import tqdm

from data.util import find_audio_files


def report_progress(progress_file, file):
    with open(progress_file, 'a', encoding='utf-8') as f:
        f.write(f'{file}\n')


def process_file(file, base_path, output_path, progress_file):
    # Hyper-parameters; feel free to adjust.
    minimum_duration = 4
    maximum_duration = 20

    # Part 1 is to split a large file into chunks.
    try:
        speech = AudioSegment.from_file(file)
    except CouldntDecodeError as e:
        print(e)
        report_progress(progress_file, file)
        return
    outdir = os.path.join(output_path, f'{os.path.relpath(file, base_path)[:-4]}').replace('.', '').strip()
    os.makedirs(outdir, exist_ok=True)
    chunks = split_on_silence(speech, min_silence_len=600, silence_thresh=-40, seek_step=100, keep_silence=50)
    for i in range(0, len(chunks)):
        if chunks[i].duration_seconds < minimum_duration or chunks[i].duration_seconds > maximum_duration:
            continue
        chunks[i].export(f"{outdir}/{i:05d}.mp3", format='mp3', parameters=["-ac", "1"])
    report_progress(progress_file, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to search for files', default='Y:\\clips\\red_rising')
    parser.add_argument('--progress_file', type=str, help='Place to store all files that have already been processed', default='Y:\\clips\\red_rising\\already_processed.txt')
    parser.add_argument('--output_path', type=str, help='Path for output files', default='Y:\\clips\\red_rising_split')
    parser.add_argument('--num_threads', type=int, help='Number of concurrent workers processing files.', default=4)
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
        list(tqdm(pool.imap(functools.partial(process_file, output_path=args.output_path, base_path=args.path, progress_file=args.progress_file), files), total=len(files)))
