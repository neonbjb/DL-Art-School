import argparse
import logging
import os
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from pydub.silence import split_on_silence
from data.util import find_audio_files
from tqdm import tqdm


# Uses pydub to process a directory of audio files, splitting them into clips at points where it detects a small amount
# of silence.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--out')
    args = parser.parse_args()
    minimum_duration = 2
    maximum_duration = 20
    files = find_audio_files(args.path, include_nonwav=True)
    for e, wav_file in enumerate(tqdm(files)):
        print(f"Processing {wav_file}..")
        outdir = os.path.join(args.out, f'{e}_{os.path.basename(wav_file[:-4])}').replace('.', '').strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            speech = AudioSegment.from_file(wav_file)
        except CouldntDecodeError as e:
            print(e)
            continue
        chunks = split_on_silence(speech, min_silence_len=400, silence_thresh=-40,
                                  seek_step=100, keep_silence=50)

        for i in range(0, len(chunks)):
            if chunks[i].duration_seconds < minimum_duration or chunks[i].duration_seconds > maximum_duration:
                continue
            chunks[i].export(f"{outdir}/{i:05d}.mp3", format='mp3', parameters=["-ac", "1"])

if __name__ == '__main__':
    main()
