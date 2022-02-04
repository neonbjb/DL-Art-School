import os
import random

import torch

from data.audio.unsupervised_audio_dataset import load_audio
from data.util import find_files_of_type, is_audio_file
from models.diffusion.gaussian_diffusion import get_named_beta_schedule
from models.diffusion.respace import SpacedDiffusion, space_timesteps
from trainer.injectors.audio_injectors import TorchMelSpectrogramInjector
from utils.audio import plot_spectrogram


def wav_to_mel(wav, mel_norms_file='../experiments/clips_mel_norms.pth'):
    """
    Converts an audio clip into a MEL tensor that the vocoder, DVAE and GptTts models use whenever a MEL is called for.
    """
    return TorchMelSpectrogramInjector({'in': 'wav', 'out': 'mel', 'mel_norm_file': mel_norms_file},{})({'wav': wav})['mel']


def convert_mel_to_codes(dvae_model, mel):
    """
    Converts an audio clip into discrete codes.
    """
    dvae_model.eval()
    with torch.no_grad():
        return dvae_model.get_codebook_indices(mel)


def load_gpt_conditioning_inputs_from_directory(path, num_candidates=3, sample_rate=22050, max_samples=44100):
    candidates = find_files_of_type('img', os.path.dirname(path), qualifier=is_audio_file)[0]
    assert len(candidates) < 50000  # Sanity check to ensure we aren't loading "related files" that aren't actually related.
    if len(candidates) == 0:
        print(f"No conditioning candidates found for {path} (not even the clip itself??)")
        raise NotImplementedError()
    # Sample with replacement. This can get repeats, but more conveniently handles situations where there are not enough candidates.
    related_mels = []
    for k in range(num_candidates):
        rel_clip = load_audio(random.choice(candidates), sample_rate)
        gap = rel_clip.shape[-1] - max_samples
        if gap > 0:
            rand_start = random.randint(0, gap)
            rel_clip = rel_clip[:, rand_start:rand_start+max_samples]
        as_mel = wav_to_mel(rel_clip)
        related_mels.append(as_mel)
    return torch.stack(related_mels, dim=0)


def load_discrete_vocoder_diffuser(trained_diffusion_steps=4000, desired_diffusion_steps=200, schedule='linear'):
    """
    Helper function to load a GaussianDiffusion instance configured for use as a vocoder.
    """
    return SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule(schedule, trained_diffusion_steps))


def do_spectrogram_diffusion(diffusion_model, dvae_model, diffuser, mel_codes, conditioning_input, spectrogram_compression_factor=128, plt_spec=False, mean=False):
    """
    Uses the specified diffusion model and DVAE model to convert the provided MEL & conditioning inputs into an audio clip.
    """
    diffusion_model.eval()
    dvae_model.eval()
    with torch.no_grad():
        mel = dvae_model.decode(mel_codes)[0]
        if plt_spec:
            plot_spectrogram(mel[0].cpu())

        # Pad MEL to multiples of 2048//spectrogram_compression_factor
        msl = mel.shape[-1]
        dsl = 2048 // spectrogram_compression_factor
        gap = dsl - (msl % dsl)
        if gap > 0:
            mel = torch.nn.functional.pad(mel, (0, gap))

        output_shape = (mel.shape[0], 1, mel.shape[-1] * spectrogram_compression_factor)
        if mean:
            return diffuser.p_sample_loop(diffusion_model, output_shape, noise=torch.zeros(output_shape, device=mel_codes.device),
                                          model_kwargs={'spectrogram': mel, 'conditioning_input': conditioning_input})
        else:
            return diffuser.p_sample_loop(diffusion_model, output_shape, model_kwargs={'spectrogram': mel, 'conditioning_input': conditioning_input})

