import os

import torch
import numpy as np
import torchaudio
import torchvision

from models.audio.music.tfdpc_v5 import TransformerDiffusionWithPointConditioning
from utils.music_utils import get_cheater_decoder
from utils.util import load_audio
from trainer.injectors.audio_injectors import TorchMelSpectrogramInjector
from trainer.injectors.audio_injectors import MusicCheaterLatentInjector
from models.diffusion.respace import SpacedDiffusion
from models.diffusion.respace import space_timesteps
from models.diffusion.gaussian_diffusion import get_named_beta_schedule
from models.audio.music.transformer_diffusion12 import TransformerDiffusionWithCheaterLatent


def join_music(clip1, clip1_cut, clip2, clip2_cut, mix_time, results_dir):
    with torch.no_grad():
        spec_fn = TorchMelSpectrogramInjector({'n_mel_channels': 256, 'mel_fmax': 11000, 'filter_length': 16000, 'true_normalization': True,
                                                    'normalize': True, 'in': 'in', 'out': 'out'}, {}).cuda()
        cheater_encoder = MusicCheaterLatentInjector({'in': 'in', 'out': 'out'}, {}).cuda()
        model = TransformerDiffusionWithPointConditioning(in_channels=256, out_channels=512, model_channels=1024,
                                                          contraction_dim=512, num_heads=8, num_layers=12, dropout=0,
                                                          use_fp16=False, unconditioned_percentage=0).eval().cuda()
        diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [256]), model_mean_type='epsilon',
                               model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 4000),
                               conditioning_free=True, conditioning_free_k=2)
        model.load_state_dict(torch.load('x:/dlas/experiments/train_music_cheater_gen_v5/models/72000_generator_ema.pth'))
        clip1 = load_audio(clip1, 22050)[:-(clip1_cut*22050)].cuda()
        clip1_mel = spec_fn({'in': clip1.unsqueeze(0)})['out']
        clip1_cheater = cheater_encoder({'in': clip1_mel})['out']
        clip2 = load_audio(clip2, 22050)[clip2_cut*22050:].cuda()
        clip2_mel = spec_fn({'in': clip2.unsqueeze(0)})['out']
        clip2_cheater = cheater_encoder({'in': clip2_mel})['out']


        blank_cheater_sz = (22050*mix_time//4096)
        sample_template = torch.cat([clip1_cheater[:,:,-25:],
                                     torch.zeros(1,256,blank_cheater_sz, device='cuda'),
                                     clip2_cheater[:,:,:25]], dim=-1)
        mask = torch.ones_like(sample_template)
        mask[:,:,25:-25] = 0

        def custom_conditioning_endpoint_fetch(cond_enc, ts):
            clip_sz = 100
            combined_cheater = torch.cat([clip1_cheater[:,:,-clip_sz:], clip2_cheater[:,:,:clip_sz]], dim=-1)
            enc = cond_enc(combined_cheater, ts)
            start_cond = enc[:,:,clip_sz-25]  # About 5 seconds back into the clip.
            end_cond = enc[:,:,clip_sz+25]
            return start_cond, end_cond


        gen_cheater = diffuser.p_sample_loop_with_guidance(model, sample_template, mask,
                                                           model_kwargs={'custom_conditioning_fetcher': custom_conditioning_endpoint_fetch})

        cheater_decoder_diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [64]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 4000),
                           conditioning_free=True, conditioning_free_k=1)
        cheater_to_mel = get_cheater_decoder().diff.cuda()
        gen_mel = cheater_decoder_diffuser.ddim_sample_loop(cheater_to_mel, (1,256,gen_cheater.shape[-1]*16), progress=True,
                                         model_kwargs={'codes': gen_cheater.permute(0,2,1)})
        torchvision.utils.save_image((gen_mel + 1)/2, f'{results_dir}/mel.png')

        from utils.music_utils import get_mel2wav_v3_model
        m2w = get_mel2wav_v3_model().cuda()
        spectral_diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [32]), model_mean_type='epsilon',
                               model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 4000),
                               conditioning_free=True, conditioning_free_k=1)
        from trainer.injectors.audio_injectors import denormalize_mel
        gen_mel_denorm = denormalize_mel(gen_mel)
        output_shape = (1,16,gen_mel_denorm.shape[-1]*256//16)
        gen_wav = spectral_diffuser.ddim_sample_loop(m2w, output_shape, progress=True, model_kwargs={'codes': gen_mel_denorm})
        from trainer.injectors.audio_injectors import pixel_shuffle_1d
        gen_wav = pixel_shuffle_1d(gen_wav, 16)

        torchaudio.save(f'{results_dir}/out.wav', gen_wav.squeeze(1).cpu(), 22050)


if __name__ == '__main__':
    results_dir = '../results/audio_joiner'
    clip1 = 'Y:\\sources\\manual_podcasts_music\\2\\The Glitch Mob - Discography\\2014 - Love, Death Immortality\\2. Our Demons (feat. Aja Volkman).mp3'
    clip1_cut = 35  # Seconds
    clip2 = 'Y:\\sources\\manual_podcasts_music\\2\\The Glitch Mob - Discography\\2014 - Love, Death Immortality\\9. Carry The Sun.mp3'
    clip2_cut = 1
    mix_time = 10
    os.makedirs(results_dir, exist_ok=True)

    join_music(clip1, clip1_cut, clip2, clip2_cut, mix_time, results_dir)