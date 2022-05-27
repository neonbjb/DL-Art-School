import os
import random

import torch

from data.audio.unsupervised_audio_dataset import load_audio
from data.util import find_files_of_type, is_audio_file
from models.audio.vocoders.univnet.generator import UnivNetGenerator
from models.diffusion.gaussian_diffusion import get_named_beta_schedule
from models.diffusion.respace import SpacedDiffusion, space_timesteps
from trainer.injectors.audio_injectors import TorchMelSpectrogramInjector, MelSpectrogramInjector
from utils.audio import plot_spectrogram
from utils.util import load_model_from_config


def load_speech_dvae():
    dvae = load_model_from_config("../experiments/train_diffusion_vocoder_22k_level.yml",
                                  "dvae").cpu()
    dvae.eval()
    return dvae


def load_univnet_vocoder():
    model = UnivNetGenerator()
    sd = torch.load('../experiments/univnet_c32_pretrained_libri.pt', map_location='cpu')
    model.load_state_dict(sd['model_g'])
    model = model.cpu()
    model.eval(inference=True)
    return model


def load_clvp():
    from models.clip.text_voice_clip import VoiceCLIP
    clvp = VoiceCLIP(dim_text=768, dim_speech=768, dim_latent=768, num_text_tokens=256, text_enc_depth=20,
                          text_seq_len=350, text_heads=12, num_speech_tokens=8192, speech_enc_depth=20,
                          speech_heads=12, speech_seq_len=430, text_mask_percentage=0, voice_mask_percentage=0,
                          use_xformers=True)
    clvp.load_state_dict(torch.load(f"../experiments/clvp_md.pth", map_location=torch.device('cpu')))
    clvp = clvp.eval()
    return clvp


def wav_to_mel(wav, mel_norms_file='../experiments/clips_mel_norms.pth'):
    """
    Converts an audio clip into a MEL tensor that the vocoder, DVAE and GptTts models use whenever a MEL is called for.
    """
    return TorchMelSpectrogramInjector({'in': 'wav', 'out': 'mel', 'mel_norm_file': mel_norms_file},{})({'wav': wav})['mel']


def wav_to_univnet_mel(wav, do_normalization=False):
    """
    Converts an audio clip into a MEL tensor that the univnet vocoder knows how to decode.
    """
    return MelSpectrogramInjector({'in': 'wav', 'out': 'mel', 'sampling_rate': 24000,
                                   'n_mel_channels': 100, 'mel_fmax': 12000,
                                   'do_normalization': do_normalization},{})({'wav': wav})['mel']


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


def load_discrete_vocoder_diffuser(trained_diffusion_steps=4000, desired_diffusion_steps=200, schedule='linear', enable_conditioning_free_guidance=False, conditioning_free_k=1):
    """
    Helper function to load a GaussianDiffusion instance configured for use as a vocoder.
    """
    return SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule(schedule, trained_diffusion_steps),
                           conditioning_free=enable_conditioning_free_guidance, conditioning_free_k=conditioning_free_k)


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

