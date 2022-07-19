import os

import torch
import numpy as np
import torchaudio
import torchvision
from tqdm import tqdm

from models.audio.music.tfdpc_v5 import TransformerDiffusionWithPointConditioning
from utils.music_utils import get_cheater_decoder, get_mel2wav_v3_model
from utils.util import load_audio
from trainer.injectors.audio_injectors import TorchMelSpectrogramInjector, denormalize_torch_mel, pixel_shuffle_1d
from trainer.injectors.audio_injectors import MusicCheaterLatentInjector
from models.diffusion.respace import SpacedDiffusion
from models.diffusion.respace import space_timesteps
from models.diffusion.gaussian_diffusion import get_named_beta_schedule


def join_music_with_cheaters(clip1_cheater, clip2_cheater, results_dir):
    clip1_leadin = clip1_cheater[:,:,-60:]
    clip1_cheater = clip1_cheater[:,:,:-60]
    clip2_leadin = clip2_cheater[:,:,:60]
    clip2_cheater = clip2_cheater[:,:,60:]

    """
    # Original model
    model = TransformerDiffusionWithPointConditioning(in_channels=256, out_channels=512, model_channels=1024,
                                                      contraction_dim=512, num_heads=8, num_layers=12, dropout=0,
                                                      use_fp16=False, unconditioned_percentage=0, time_proj=True).eval().cuda()
    model.load_state_dict(torch.load('x:/dlas/experiments/train_music_cheater_gen_v5/models/206000_generator_ema.pth'))
    diffusion_type = 'linear'
    """
    model = TransformerDiffusionWithPointConditioning(in_channels=256, out_channels=512, model_channels=1024,
                                                      contraction_dim=512, num_heads=8, num_layers=32, dropout=0,
                                                      use_fp16=False, unconditioned_percentage=0, time_proj=False,
                                                      new_cond=True, regularization=False).eval().cuda()
    model.load_state_dict(torch.load('x:/dlas/experiments/train_music_cheater_gen_v5_cosine_40_lyr/models/64000_generator_ema.pth'))
    diffusion_type = 'cosine'
    diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [256]), model_mean_type='epsilon',
                               model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule(diffusion_type, 4000),
                               conditioning_free=True, conditioning_free_k=2)

    inp = torch.cat([clip1_leadin, torch.zeros(1, 256, 240, device='cuda'), clip2_leadin], dim=-1)
    mask = torch.ones_like(inp)
    mask[:, :, 60:-60] = 0
    gen_cheater = diffuser.ddim_sample_loop_with_guidance(model, inp, mask,  # causal=True, causal_slope=4,
                                                          model_kwargs={'cond_left': clip1_cheater,
                                                                        'cond_right': clip2_cheater})

    cheater_to_mel = get_cheater_decoder().diff.cuda()
    cheater_decoder_diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [64]), model_mean_type='epsilon',
                                               model_var_type='learned_range', loss_type='mse',
                                               betas=get_named_beta_schedule('linear', 4000),
                                               conditioning_free=True, conditioning_free_k=1)
    m2w = get_mel2wav_v3_model().cuda()
    spectral_diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [32]), model_mean_type='epsilon',
                                        model_var_type='learned_range', loss_type='mse',
                                        betas=get_named_beta_schedule('linear', 4000),
                                        conditioning_free=True, conditioning_free_k=1)

    MAX_CONTEXT = 30 * 22050 // 4096
    chunks = torch.split(gen_cheater, MAX_CONTEXT, dim=-1)
    gen_wavs = []
    for i, chunk_cheater in enumerate(tqdm(chunks)):
        gen_mel = cheater_decoder_diffuser.ddim_sample_loop(cheater_to_mel, (1, 256, chunk_cheater.shape[-1] * 16),
                                                            progress=True,
                                                            model_kwargs={'codes': chunk_cheater.permute(0, 2, 1)})
        torchvision.utils.save_image((gen_mel + 1) / 2, f'{results_dir}/mel_{i}.png')

        gen_mel_denorm = denormalize_torch_mel(gen_mel)
        output_shape = (1, 16, gen_mel_denorm.shape[-1] * 256 // 16)
        wav = spectral_diffuser.ddim_sample_loop(m2w, output_shape, progress=True,
                                                 model_kwargs={'codes': gen_mel_denorm})
        gen_wavs.append(pixel_shuffle_1d(wav, 16))

    gen_wav = torch.cat(gen_wavs, dim=-1)
    torchaudio.save(f'{results_dir}/out.wav', gen_wav.squeeze(1).cpu(), 22050)


def join_music(clip1, clip2, results_dir):
    with torch.no_grad():
        spec_fn = TorchMelSpectrogramInjector({'n_mel_channels': 256, 'mel_fmax': 11000, 'filter_length': 16000, 'true_normalization': True,
                                                    'normalize': True, 'in': 'in', 'out': 'out'}, {}).cuda()
        cheater_encoder = MusicCheaterLatentInjector({'in': 'in', 'out': 'out'}, {}).cuda()
        clip1 = load_audio(clip1, 22050).cuda()
        clip1_mel = spec_fn({'in': clip1.unsqueeze(0)})['out']
        clip1_cheater = cheater_encoder({'in': clip1_mel})['out']
        clip2 = load_audio(clip2, 22050).cuda()
        clip2_mel = spec_fn({'in': clip2.unsqueeze(0)})['out']
        clip2_cheater = cheater_encoder({'in': clip2_mel})['out']
        join_music_with_cheaters(clip1_cheater, clip2_cheater, results_dir)



if __name__ == '__main__':
    """
    things_to_try = {
        'goo': ('Y:\\separated\\bt-music-5\\[2002] Gutterflower\\02 - Think About Me', 0),
        'sm1': ('Y:\\separated\\silk\\MonstercatSilkShowcase\\910', 79),
        'sm2': ('Y:\\separated\\silk\\MonstercatSilkShowcase\\1025', 105),
        'sm3': ('Y:\\separated\\silk\\MonstercatSilkShowcase\\1026', 43),
        'sm4': ('Y:\\separated\\silk\\MonstercatSilkShowcase\\1026', 77),
        'sm5': ('Y:\\separated\\silk\\MonstercatSilkShowcase\\1027', 8),
        'sm6': ('Y:\\separated\\silk\\MonstercatSilkShowcase\\1027', 28),
        'sm6': ('Y:\\separated\\silk\\MonstercatSilkShowcase\\1017', 90),
        'tron': ('Y:\\separated\\bt-music-2\\2011 - TRON Legacy - Translucence (EP) - (320 kbps)\\01 Derezzed', 0),
        'lateralus': ('Y:\\separated\\bt-music-2\\Lateralus\\09 - lateralus\\00011', 11),
        'streets_have_no_name': ('Y:\\separated\\bt-music-2\\U2 - (1987) The Joshua Tree\\01-Where The Streets Have No Name', 1),
        'shinra': ('Y:\\separated\\bt-music-1\\final_fantasy_vii_soundtrack\\20-Infiltrating Shinra Tower', 1),
        'bombing_run': ('Y:\\separated\\bt-music-1\\final_fantasy_vii_soundtrack\\02-Opening ~ Bombing Mission', 2),
        'machine_gun': ('Y:\\separated\\bt-music-1\\ff8-fithos_lusec_wecos_vinosec-1999\\08 - The Man with the Machine Gun', 2),
    }
    for k, v in things_to_try.items():
        results_dir = f'../results/audio_joiner/{k}'
        src_path, start = v
        clip1 = f'{src_path}\\{start:05d}\\no_vocals.wav'
        clip2 = f'{src_path}\\{(start+2):05d}\\no_vocals.wav'
        os.makedirs(results_dir, exist_ok=True)
        join_music(clip1, clip2, results_dir)
    """
    results_dir = f'../results/audio_joiner/machine_gun_from_cheater'
    os.makedirs(results_dir, exist_ok=True)
    cheater = torch.tensor(np.load('Y:\\separated\\large_mel_cheaters\\bt-music-1\\ff8-fithos_lusec_wecos_vinosec-1999\\08 - The Man with the Machine Gun\\0.npz')['arr_0']).cuda()
    join_music_with_cheaters(cheater[:,:,:230], cheater[:,:,-230:], results_dir)
