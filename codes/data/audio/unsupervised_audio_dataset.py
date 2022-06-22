import os
import random
import sys

import torch
import torch.utils.data
import torch.nn.functional as F
import torchaudio
from audio2numpy import open_audio
from tqdm import tqdm

from data.util import find_files_of_type, is_audio_file, load_paths_from_cache
from models.audio.tts.tacotron2.taco_utils import load_wav_to_torch
from utils.util import opt_get


def load_audio(audiopath, sampling_rate):
    if audiopath[-4:] == '.wav':
        audio, lsr = load_wav_to_torch(audiopath)
    elif audiopath[-4:] == '.mp3':
        # https://github.com/neonbjb/pyfastmp3decoder  - Definitely worth it.
        from pyfastmp3decoder.mp3decoder import load_mp3
        audio, lsr = load_mp3(audiopath, sampling_rate)
        audio = torch.FloatTensor(audio)
    else:
        audio, lsr = open_audio(audiopath)
        audio = torch.FloatTensor(audio)

    # Remove any channel data.
    if len(audio.shape) > 1:
        if audio.shape[0] < 5:
            audio = audio[0]
        else:
            assert audio.shape[1] < 5
            audio = audio[:, 0]

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '10' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 10) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1, 1)

    return audio.unsqueeze(0)


def load_similar_clips(path, sample_length, sample_rate, n=3, fallback_to_self=True):
    sim_path = os.path.join(os.path.dirname(path), 'similarities.pth')
    candidates = []
    if os.path.exists(sim_path):
        similarities = torch.load(sim_path)
        fname = os.path.basename(path)
        if fname in similarities.keys():
            candidates = [os.path.join(os.path.dirname(path), s) for s in similarities[fname]]
        else:
            print(f'Similarities list found for {path} but {fname} was not in that list.')
        #candidates.append(path)  # Always include self as a possible similar clip.
    if len(candidates) == 0:
        if fallback_to_self:
            candidates = [path]
        else:
            candidates = find_files_of_type('img', os.path.dirname(path), qualifier=is_audio_file)[0]

    assert len(candidates) < 50000  # Sanity check to ensure we aren't loading "related files" that aren't actually related.
    if len(candidates) == 0:
        print(f"No conditioning candidates found for {path}")
        raise NotImplementedError()

    # Sample with replacement. This can get repeats, but more conveniently handles situations where there are not enough candidates.
    related_clips = []
    contains_self = False
    for k in range(n):
        rel_path = random.choice(candidates)
        contains_self = contains_self or (rel_path == path)
        rel_clip = load_audio(rel_path, sample_rate)
        gap = rel_clip.shape[-1] - sample_length
        if gap < 0:
            rel_clip = F.pad(rel_clip, pad=(0, abs(gap)))
        elif gap > 0:
            rand_start = random.randint(0, gap)
            rel_clip = rel_clip[:, rand_start:rand_start+sample_length]
        related_clips.append(rel_clip)
    if n > 1:
        return torch.stack(related_clips, dim=0), contains_self
    else:
        return related_clips[0], contains_self


class UnsupervisedAudioDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        path = opt['path']
        cache_path = opt['cache_path']  # Will fail when multiple paths specified, must be specified in this case.
        exclusions = []
        if 'exclusions' in opt.keys():
            for exc in opt['exclusions']:
                with open(exc, 'r') as f:
                    exclusions.extend(f.read().splitlines())
        ew = opt_get(opt, ['endswith'], [])
        assert isinstance(ew, list)
        not_ew = opt_get(opt, ['not_endswith'], [])
        assert isinstance(not_ew, list)
        self.audiopaths = load_paths_from_cache(path, cache_path, exclusions, endswith=ew, not_endswith=not_ew)

        # Parse options
        self.sampling_rate = opt_get(opt, ['sampling_rate'], 22050)
        self.pad_to = opt_get(opt, ['pad_to_seconds'], None)
        if self.pad_to is not None:
            self.pad_to *= self.sampling_rate
        self.pad_to = opt_get(opt, ['pad_to_samples'], self.pad_to)
        self.min_length = opt_get(opt, ['min_length'], 0)
        self.dont_clip = opt_get(opt, ['dont_clip'], False)

        # "Resampled clip" is audio data pulled from the basis of "clip" but with randomly different bounds. There are no
        # guarantees that "clip_resampled" is different from "clip": in fact, if "clip" is less than pad_to_seconds/samples,
        self.should_resample_clip = opt_get(opt, ['resample_clip'], False)

        # "Extra samples" are other audio clips pulled from wav files in the same directory as the 'clip' wav file.
        self.extra_samples = opt_get(opt, ['extra_samples'], 0)
        self.extra_sample_len = opt_get(opt, ['extra_sample_length'], 44000)

        self.debug_loading_failures = opt_get(opt, ['debug_loading_failures'], True)

    def get_audio_for_index(self, index):
        audiopath = self.audiopaths[index]
        audio = load_audio(audiopath, self.sampling_rate)
        assert audio.shape[1] > self.min_length
        if self.dont_clip:
            assert audio.shape[1] <= self.pad_to
        return audio, audiopath

    def get_related_audio_for_index(self, index):
        if self.extra_samples <= 0:
            return None, 0
        audiopath = self.audiopaths[index]
        return load_similar_clips(audiopath, self.extra_sample_len, self.sampling_rate, n=self.extra_samples)

    def __getitem__(self, index):
        try:
            # Split audio_norm into two tensors of equal size.
            audio_norm, filename = self.get_audio_for_index(index)
            alt_files, alt_is_self = self.get_related_audio_for_index(index)
        except:
            if self.debug_loading_failures:
                print(f"Error loading audio for file {self.audiopaths[index]} {sys.exc_info()}")
            return self[random.randint(0,len(self))]

        # When generating resampled clips, skew is a bias that tries to spread them out from each other, reducing their
        # influence on one another.
        skew = [-1, 1] if self.should_resample_clip else [0]
        # To increase variability, which skew is applied to the clip and resampled_clip is randomized.
        random.shuffle(skew)
        clips = []
        prepad_length = min(audio_norm.shape[-1], self.pad_to)
        for sk in skew:
            if self.pad_to is not None:
                if audio_norm.shape[-1] <= self.pad_to:
                    clips.append(torch.nn.functional.pad(audio_norm, (0, self.pad_to - audio_norm.shape[-1])))
                else:
                    gap = audio_norm.shape[-1] - self.pad_to
                    start = min(max(random.randint(0, gap-1) + sk * gap // 2, 0), gap-1)
                    clips.append(audio_norm[:, start:start+self.pad_to])
            else:
                clips.append(audio_norm)

        output = {
            'prepad_length': prepad_length,
            'clip': clips[0],
            'clip_lengths': torch.tensor(clips[0].shape[-1]),
            'path': filename,
        }
        if self.should_resample_clip:
            output['resampled_clip'] = clips[1]
        if self.extra_samples > 0:
            output['alt_clips'] = alt_files
            output['alt_contains_self'] = alt_is_self
        return output

    def __len__(self):
        return len(self.audiopaths)


if __name__ == '__main__':
    params = {
        'mode': 'unsupervised_audio',
        'path': ['Y:\\separated\\yt-music-0', 'Y:\\separated\\yt-music-1',
                 'Y:\\separated\\bt-music-1', 'Y:\\separated\\bt-music-2',
                 'Y:\\separated\\bt-music-3', 'Y:\\separated\\bt-music-4',
                 'Y:\\separated\\bt-music-5'],
        'cache_path': 'Y:\\separated\\no-vocals-cache-win.pth',
        'endswith': ['no_vocals.wav'],
        'sampling_rate': 22050,
        'pad_to_samples': 200000,
        'resample_clip': False,
        'extra_samples': 1,
        'extra_sample_length': 100000,
        'phase': 'train',
        'n_workers': 1,
        'batch_size': 16,
    }
    from data import create_dataset, create_dataloader

    ds = create_dataset(params)
    dl = create_dataloader(ds, params)
    i = 0
    for b in tqdm(dl):
        for b_ in range(b['clip'].shape[0]):
            #pass
            torchaudio.save(f'{i}_clip_{b_}.wav', b['clip'][b_], ds.sampling_rate)
            torchaudio.save(f'{i}_alt_clip_{b_}.wav', b['alt_clips'][b_], ds.sampling_rate)
            i += 1
        if i > 200:
            break
