import argparse
import random

import torch
import torch.nn.functional as F
import torchaudio
import yaml

from data.audio.unsupervised_audio_dataset import load_audio
from data.util import is_audio_file, find_files_of_type
from models.tacotron2.text import text_to_sequence
from scripts.audio.gen.speech_synthesis_utils import do_spectrogram_diffusion, \
    load_discrete_vocoder_diffuser, wav_to_mel
from trainer.injectors.base_injectors import TorchMelSpectrogramInjector
from utils.options import Loader
from utils.util import load_model_from_config


def do_vocoding(dvae, vocoder, diffuser, codes, cond=None, plot_spec=False):
    return


# Loads multiple conditioning files at random from a folder.
def load_conditioning_candidates(path, num_conds, sample_rate=22050, cond_length=44100):
    candidates = find_files_of_type('img', path, qualifier=is_audio_file)[0]
    # Sample with replacement. This can get repeats, but more conveniently handles situations where there are not enough candidates.
    related_mels = []
    for k in range(num_conds):
        rel_clip = load_audio(candidates[k], sample_rate)
        gap = rel_clip.shape[-1] - cond_length
        if gap < 0:
            rel_clip = F.pad(rel_clip, pad=(0, abs(gap)))
        elif gap > 0:
            rand_start = random.randint(0, gap)
            rel_clip = rel_clip[:, rand_start:rand_start + cond_length]
        mel_clip = wav_to_mel(rel_clip.unsqueeze(0)).squeeze(0)
        related_mels.append(mel_clip)
    return torch.stack(related_mels, dim=0).unsqueeze(0).cuda(), rel_clip.unsqueeze(0).cuda()


def load_conditioning(path, sample_rate=22050, cond_length=44100):
    rel_clip = load_audio(path, sample_rate)
    gap = rel_clip.shape[-1] - cond_length
    if gap < 0:
        rel_clip = F.pad(rel_clip, pad=(0, abs(gap)))
    elif gap > 0:
        rand_start = random.randint(0, gap)
        rel_clip = rel_clip[:, rand_start:rand_start + cond_length]
    mel_clip = wav_to_mel(rel_clip.unsqueeze(0)).squeeze(0)
    return mel_clip.unsqueeze(0).cuda(), rel_clip.unsqueeze(0).cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt_diffuse', type=str, help='Path to options YAML file used to train the diffusion model', default='X:\\dlas\\experiments\\train_diffusion_vocoder_with_cond_new_dvae.yml')
    parser.add_argument('-diffusion_model_name', type=str, help='Name of the diffusion model in opt.', default='generator')
    parser.add_argument('-diffusion_model_path', type=str, help='Diffusion model checkpoint to load.', default='X:\\dlas\\experiments\\train_diffusion_vocoder_with_cond_new_dvae_full\\models\\6100_generator_ema.pth')
    parser.add_argument('-dvae_model_name', type=str, help='Name of the DVAE model in opt.', default='dvae')
    parser.add_argument('-opt_gpt_tts', type=str, help='Path to options YAML file used to train the GPT-TTS model', default='X:\\dlas\\experiments\\train_gpt_tts.yml')
    parser.add_argument('-gpt_tts_model_name', type=str, help='Name of the GPT TTS model in opt.', default='gpt')
    parser.add_argument('-gpt_tts_model_path', type=str, help='GPT TTS model checkpoint to load.', default='X:\\dlas\\experiments\\train_gpt_tts_no_pos\\models\\28500_gpt_ema.pth')
    parser.add_argument('-text', type=str, help='Text to speak.', default="Please set this in the courier drone when we dock.")
    parser.add_argument('-cond_path', type=str, help='Path to condioning sample.', default='Y:\\clips\\books1\\754_Dan Simmons - The Rise Of Endymion 356 of 450\\00026.wav')
    args = parser.parse_args()

    print("Loading GPT TTS..")
    with open(args.opt_gpt_tts, mode='r') as f:
        gpt_opt = yaml.load(f, Loader=Loader)
    gpt_opt['networks'][args.gpt_tts_model_name]['kwargs']['checkpointing'] = False  # Required for beam search
    gpt = load_model_from_config(preloaded_options=gpt_opt, model_name=args.gpt_tts_model_name, also_load_savepoint=False, load_path=args.gpt_tts_model_path)

    print("Loading data..")
    text = torch.IntTensor(text_to_sequence(args.text, ['english_cleaners'])).unsqueeze(0).cuda()
    conds, cond_wav = load_conditioning(args.cond_path)

    print("Performing GPT inference..")
    codes = gpt.inference(text, conds, num_beams=32, repetition_penalty=10.0)

    # Delete the GPT TTS model to free up GPU memory
    del gpt

    print("Loading DVAE..")
    dvae = load_model_from_config(args.opt_diffuse, args.dvae_model_name)
    print("Loading Diffusion Model..")
    diffusion = load_model_from_config(args.opt_diffuse, args.diffusion_model_name, also_load_savepoint=False, load_path=args.diffusion_model_path)
    diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=50)

    print("Performing vocoding..")
    wav = do_spectrogram_diffusion(diffusion, dvae, diffuser, codes, cond_wav, spectrogram_compression_factor=128, plt_spec=False)
    torchaudio.save('gpt_tts_output.wav', wav.squeeze(0).cpu(), 10025)