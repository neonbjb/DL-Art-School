# Combines all libriTTS WAV->text mappings into a single file
import os
import random

import audio2numpy
import torch
from scipy.io import wavfile
from tqdm import tqdm

from utils.audio_resampler import AudioResampler


def secs_to_frames(secs, sr):
    return int(secs*sr)


def get_audio_clip(audio, sr, start, end):
    start = secs_to_frames(start, sr)
    end = secs_to_frames(end, sr)
    assert end > start
    if end >= audio.shape[0]:
        return None
    return audio[start:end]


# Produces an audio clip that would produce a MEL spectrogram of length mel_length by parsing parsed_sentences starting
# at starting_index and moving forwards until the full length is finished.
# Returns:
#  On failure, returns tuple: (end_index, None, [], [])
#  On success: returns tuple: (end_index, clip, start_points, end_points)
#    clip.shape = (<mel_length*256>,)
#    start_points = list(ints) where each sentence in the clip starts
#    end_points = list(ints) where each sentence in the clip ends
def gather_clip(audio, parsed_sentences, starting_index, sr, mel_length):
    audio_length = (mel_length * 256) / sr  # This is technically a hyperparameter, but I have no intent of changing the MEL hop length.
    starts = []
    ends = []
    start, end = parsed_sentences[starting_index][4:6]
    start = float(start)
    end = float(end)
    clipstart = max(start - random.random() * 2, 0)  # Offset start backwards by up to 2 seconds
    clipend = start + audio_length
    clip = get_audio_clip(audio, sr, clipstart, clipend)
    if clip is not None:
        # Fetch the start and endpoints that go along with this clip.
        starts.append(secs_to_frames(start-clipstart, sr))
        while end < clipend:
            ends.append(secs_to_frames(end-clipstart, sr))
            starting_index += 1
            if starting_index >= len(parsed_sentences):
                break
            start, end = parsed_sentences[starting_index][4:6]
            start = float(start)
            end = float(end)
            if start < clipend:
                starts.append(secs_to_frames(start-clipstart, sr))

    return starting_index+1, clip, starts, ends


if __name__ == '__main__':
    full_book_root = 'D:\\data\\audio\\libritts\\full_books\\mp3'
    libri_root = 'D:\\data\\audio\\libritts\\test-clean'
    desired_mel_length = 2000
    desired_audio_sample_rate = 22050
    output_dir = 'D:\\data\\audio\\libritts\\stop_dataset_eval'

    os.makedirs(output_dir, exist_ok=True)
    j = 0
    readers = os.listdir(libri_root)
    for it, reader_dir in enumerate(tqdm(readers)):
        #if it <= 145:  # Hey idiot! If you change this, change j too!
        #    continue
        reader = os.path.join(libri_root, reader_dir)
        if not os.path.isdir(reader):
            continue
        for chapter_dir in os.listdir(reader):
            chapter = os.path.join(reader, chapter_dir)
            if not os.path.isdir(chapter):
                continue
            id = f'{os.path.basename(reader)}_{os.path.basename(chapter)}'
            book_file = os.path.join(chapter, f'{id}.book.tsv')
            if not os.path.exists(book_file):
                continue
            with open(book_file, encoding='utf-8') as f:
                full_chapter, sr = audio2numpy.open_audio(os.path.join(full_book_root, reader_dir, chapter_dir, f'{chapter_dir}.mp3'))
                full_chapter = torch.tensor(full_chapter)
                if len(full_chapter.shape) > 1:
                    full_chapter = full_chapter[:, 0]  # Only use mono-audio.
                resampler = AudioResampler(sr, desired_audio_sample_rate, dtype=torch.float)
                full_chapter = resampler(full_chapter.unsqueeze(0)).squeeze(0)
                parsed_sentences = [line.strip().split('\t') for line in f]
                i = 0
                while i < len(parsed_sentences):
                    i, clip, ns, ne = gather_clip(full_chapter, parsed_sentences, i, desired_audio_sample_rate, desired_mel_length)
                    if clip is not None:
                        wavfile.write(os.path.join(output_dir, f'{j}.wav'), desired_audio_sample_rate, clip.cpu().numpy())
                        torch.save((ns,ne), os.path.join(output_dir, f'{j}_se.pth'))
                        j += 1
