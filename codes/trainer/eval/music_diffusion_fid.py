import functools
import os
import os.path as osp
from glob import glob
from random import shuffle
from time import time

import numpy as np
import torch
import torchaudio
import torchvision
from pytorch_fid.fid_score import calculate_frechet_distance
from torch import distributed
from tqdm import tqdm

import trainer.eval.evaluator as evaluator
from data.audio.unsupervised_audio_dataset import load_audio
from models.audio.mel2vec import ContrastiveTrainingWrapper
from models.audio.music.unet_diffusion_waveform_gen import DiffusionWaveformGen
from models.clip.contrastive_audio import ContrastiveAudio
from models.diffusion.gaussian_diffusion import get_named_beta_schedule
from models.diffusion.respace import space_timesteps, SpacedDiffusion
from trainer.injectors.audio_injectors import denormalize_mel, TorchMelSpectrogramInjector, pixel_shuffle_1d, \
    normalize_mel, KmeansQuantizerInjector
from utils.music_utils import get_music_codegen, get_mel2wav_model, get_cheater_decoder, get_cheater_encoder, \
    get_mel2wav_v3_model, get_ar_prior
from utils.util import opt_get, load_model_from_config


class MusicDiffusionFid(evaluator.Evaluator):
    """
    Evaluator produces generate from a music diffusion model.
    """
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env, uses_all_ddp=True)
        self.real_path = opt_eval['path']
        self.data = self.load_data(self.real_path)
        self.clip = opt_get(opt_eval, ['clip_audio'], True)  # Recommend setting true for more efficient eval passes.
        self.ddim = opt_get(opt_eval, ['use_ddim'], False)
        self.causal = opt_get(opt_eval, ['causal'], False)
        self.causal_slope = opt_get(opt_eval, ['causal_slope'], 1)
        if distributed.is_initialized() and distributed.get_world_size() > 1:
            self.skip = distributed.get_world_size()  # One batch element per GPU.
        else:
            self.skip = 1
        diffusion_steps = opt_get(opt_eval, ['diffusion_steps'], 50)
        diffusion_schedule = opt_get(env['opt'], ['steps', 'generator', 'injectors', 'diffusion', 'beta_schedule', 'schedule_name'], None)
        if diffusion_schedule is None:
            print("Unable to infer diffusion schedule from master options. Getting it from eval (or guessing).")
            diffusion_schedule = opt_get(opt_eval, ['diffusion_schedule'], 'linear')
        conditioning_free_diffusion_enabled = opt_get(opt_eval, ['conditioning_free'], False)
        conditioning_free_k = opt_get(opt_eval, ['conditioning_free_k'], 1)
        self.diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [diffusion_steps]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule(diffusion_schedule, 4000),
                           conditioning_free=conditioning_free_diffusion_enabled, conditioning_free_k=conditioning_free_k)
        self.spectral_diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [16 if self.ddim else 100]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 4000),
                           conditioning_free=False, conditioning_free_k=1)
        self.dev = self.env['device']
        mode = opt_get(opt_eval, ['diffusion_type'], 'spec_decode')

        self.projector = ContrastiveAudio(model_dim=512, transformer_heads=8, dropout=0, encoder_depth=8, mel_channels=256)
        self.projector.load_state_dict(torch.load('../experiments/music_eval_projector.pth', map_location=torch.device('cpu')))

        self.local_modules = {'projector': self.projector}
        if mode == 'spec_decode':
            self.diffusion_fn = self.perform_diffusion_spec_decode
            self.squeeze_ratio = opt_eval['squeeze_ratio']
        elif 'from_codes' == mode:
            self.diffusion_fn = self.perform_diffusion_from_codes
            self.local_modules['codegen'] = get_music_codegen()
        elif 'from_codes_quant' == mode:
            self.diffusion_fn = self.perform_diffusion_from_codes_quant
        elif 'partial_from_codes_quant' == mode:
            self.diffusion_fn = functools.partial(self.perform_partial_diffusion_from_codes_quant,
                                                  partial_low=opt_eval['partial_low'],
                                                  partial_high=opt_eval['partial_high'])
        elif 'from_codes_quant_gradual_decode' == mode:
            self.diffusion_fn = self.perform_diffusion_from_codes_quant_gradual_decode
        elif 'cheater_gen' == mode:
            self.diffusion_fn = self.perform_reconstruction_from_cheater_gen
            self.local_modules['cheater_encoder'] = get_cheater_encoder()
            self.local_modules['cheater_decoder'] = get_cheater_decoder()
            self.cheater_decoder_diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [32]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 4000),
                           conditioning_free=True, conditioning_free_k=1)
            self.spectral_diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [16]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 4000),
                           conditioning_free=False, conditioning_free_k=1)
            self.spec_decoder = get_mel2wav_v3_model()  # The only reason the other functions don't use v3 is because earlier models were trained with v1 and I want to keep metrics consistent.
            self.local_modules['spec_decoder'] = self.spec_decoder
        elif 'from_ar_prior' == mode:
            self.diffusion_fn = self.perform_diffusion_from_codes_ar_prior
            self.local_modules['cheater_encoder'] = get_cheater_encoder()
            self.local_modules['cheater_decoder'] = get_cheater_decoder()
            self.cheater_decoder_diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [32]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 4000),
                           conditioning_free=True, conditioning_free_k=1)
            self.kmeans_inj = KmeansQuantizerInjector({'centroids': '../experiments/music_k_means_centroids.pth', 'in': 'in', 'out': 'out'}, {})
            self.local_modules['ar_prior'] = get_ar_prior()
            self.spec_decoder = get_mel2wav_v3_model()
            self.local_modules['spec_decoder'] = self.spec_decoder
        if not hasattr(self, 'spec_decoder'):
            self.spec_decoder = get_mel2wav_model()
            self.local_modules['spec_decoder'] = self.spec_decoder
        self.spec_fn = TorchMelSpectrogramInjector({'n_mel_channels': 256, 'mel_fmax': 11000, 'filter_length': 16000,
                                                    'normalize': True, 'in': 'in', 'out': 'out'}, {})

    def load_data(self, path):
        return list(glob(f'{path}/*.wav'))

    def perform_diffusion_spec_decode(self, audio, sample_rate=22050):
        real_resampled = audio
        audio = audio.unsqueeze(0)
        output_shape = (1, self.squeeze_ratio, audio.shape[-1] // self.squeeze_ratio)
        mel = self.spec_fn({'in': audio})['out']
        gen = self.diffuser.p_sample_loop(self.model, output_shape,
                                          model_kwargs={'codes': mel})
        gen = pixel_shuffle_1d(gen, self.squeeze_ratio)

        return gen, real_resampled, normalize_mel(self.spec_fn({'in': gen})['out']), normalize_mel(mel), sample_rate

    def perform_diffusion_from_codes(self, audio, sample_rate=22050):
        real_resampled = audio
        audio = audio.unsqueeze(0)

        mel = self.spec_fn({'in': audio})['out']
        codegen = self.local_modules['codegen'].to(mel.device)
        codes = codegen.get_codes(mel, project=True)
        mel_norm = normalize_mel(mel)
        gen_mel = self.diffuser.p_sample_loop(self.model, mel_norm.shape,
                                              model_kwargs={'codes': codes, 'conditioning_input': torch.zeros_like(mel_norm[:,:,:390])})

        gen_mel_denorm = denormalize_mel(gen_mel)
        output_shape = (1,16,audio.shape[-1]//16)
        self.spec_decoder = self.spec_decoder.to(audio.device)
        gen_wav = self.spectral_diffuser.p_sample_loop(self.spec_decoder, output_shape,
                                              model_kwargs={'aligned_conditioning': gen_mel_denorm})
        gen_wav = pixel_shuffle_1d(gen_wav, 16)

        return gen_wav, real_resampled, gen_mel, mel_norm, sample_rate

    def perform_diffusion_from_codes_quant(self, audio, sample_rate=22050):
        real_resampled = audio
        audio = audio.unsqueeze(0)

        mel = self.spec_fn({'in': audio})['out']
        mel_norm = normalize_mel(mel)
        #def denoising_fn(x):
        #    q9 = torch.quantile(x, q=.95, dim=-1).unsqueeze(-1)
        #    s = q9.clamp(1, 9999999999)
        #    x = x.clamp(-s, s) / s
        #    return x
        gen_mel = self.diffuser.p_sample_loop(self.model, mel_norm.shape, #denoised_fn=denoising_fn, clip_denoised=False,
                                              model_kwargs={'truth_mel': mel_norm,
                                                            'conditioning_input': mel_norm,
                                                            'disable_diversity': True})

        gen_mel_denorm = denormalize_mel(gen_mel)
        output_shape = (1,16,audio.shape[-1]//16)
        self.spec_decoder = self.spec_decoder.to(audio.device)
        gen_wav = self.spectral_diffuser.p_sample_loop(self.spec_decoder, output_shape,
                                              model_kwargs={'aligned_conditioning': gen_mel_denorm})
        gen_wav = pixel_shuffle_1d(gen_wav, 16)

        real_wav = self.spectral_diffuser.p_sample_loop(self.spec_decoder, output_shape,
                                              model_kwargs={'aligned_conditioning': mel})
        real_wav = pixel_shuffle_1d(real_wav, 16)

        return gen_wav, real_wav.squeeze(0), gen_mel, mel_norm, sample_rate

    def perform_partial_diffusion_from_codes_quant(self, audio, sample_rate=22050, partial_low=0, partial_high=256):
        real_resampled = audio
        audio = audio.unsqueeze(0)

        mel = self.spec_fn({'in': audio})['out']
        mel_norm = normalize_mel(mel)
        mask = torch.ones_like(mel_norm)
        mask[:, partial_low:partial_high] = 0  # This is the channel region that the model will predict.
        gen_mel = self.diffuser.p_sample_loop_with_guidance(self.model,
                                              guidance_input=mel_norm, mask=mask,
                                              model_kwargs={'truth_mel': mel,
                                                            'conditioning_input': torch.zeros_like(mel_norm[:,:,:390]),
                                                            'disable_diversity': True})

        gen_mel_denorm = denormalize_mel(gen_mel)
        output_shape = (1,16,audio.shape[-1]//16)
        self.spec_decoder = self.spec_decoder.to(audio.device)
        gen_wav = self.spectral_diffuser.p_sample_loop(self.spec_decoder, output_shape,
                                              model_kwargs={'aligned_conditioning': gen_mel_denorm})
        gen_wav = pixel_shuffle_1d(gen_wav, 16)

        return gen_wav, real_resampled, gen_mel, mel_norm, sample_rate

    def perform_diffusion_from_codes_quant_gradual_decode(self, audio, sample_rate=22050):
        real_resampled = audio
        audio = audio.unsqueeze(0)

        mel = self.spec_fn({'in': audio})['out']
        mel_norm = normalize_mel(mel)
        guidance = torch.zeros_like(mel_norm)
        mask = torch.zeros_like(mel_norm)
        GRADS = 4
        for k in range(GRADS):
            gen_mel = self.diffuser.p_sample_loop_with_guidance(self.model,
                                                                guidance_input=guidance, mask=mask,
                                                                model_kwargs={'truth_mel': mel,
                                                                              'conditioning_input': torch.zeros_like(mel_norm[:,:,:390]),
                                                                              'disable_diversity': True})
            pk = int(k*(mel_norm.shape[1]/GRADS))
            ek = int((k+1)*(mel_norm.shape[1]/GRADS))
            guidance[:, pk:ek] = gen_mel[:, pk:ek]
            mask[:, :ek] = 1

        gen_mel_denorm = denormalize_mel(gen_mel)
        output_shape = (1,16,audio.shape[-1]//16)
        self.spec_decoder = self.spec_decoder.to(audio.device)
        gen_wav = self.diffuser.p_sample_loop(self.spec_decoder, output_shape,
                                              model_kwargs={'aligned_conditioning': gen_mel_denorm})
        gen_wav = pixel_shuffle_1d(gen_wav, 16)

        return gen_wav, real_resampled, gen_mel, mel_norm, sample_rate

    def perform_reconstruction_from_cheater_gen(self, audio, sample_rate=22050):
        audio = audio.unsqueeze(0)

        mel = self.spec_fn({'in': audio})['out']
        mel_norm = normalize_mel(mel)
        cheater = self.local_modules['cheater_encoder'].to(audio.device)(mel_norm)

        # 1. Generate the cheater latent using the input as a reference.
        sampler = self.diffuser.ddim_sample_loop if self.ddim else self.diffuser.p_sample_loop
        gen_cheater = sampler(self.model, cheater.shape, progress=True,
                              causal=self.causal, causal_slope=self.causal_slope,
                              model_kwargs={'conditioning_input': cheater})

        # 2. Decode the cheater into a MEL
        gen_mel = self.cheater_decoder_diffuser.ddim_sample_loop(self.local_modules['cheater_decoder'].diff.to(audio.device), (1,256,gen_cheater.shape[-1]*16), progress=True,
                                                 model_kwargs={'codes': gen_cheater.permute(0,2,1)})

        # 3. And then the MEL back into a spectrogram
        output_shape = (1,16,audio.shape[-1]//16)
        self.spec_decoder = self.spec_decoder.to(audio.device)
        gen_mel_denorm = denormalize_mel(gen_mel)
        gen_wav = self.spectral_diffuser.ddim_sample_loop(self.spec_decoder, output_shape,
                                              model_kwargs={'codes': gen_mel_denorm})
        gen_wav = pixel_shuffle_1d(gen_wav, 16)

        real_wav = self.spectral_diffuser.ddim_sample_loop(self.spec_decoder, output_shape,
                                              model_kwargs={'codes': mel})
        real_wav = pixel_shuffle_1d(real_wav, 16)

        return gen_wav, real_wav.squeeze(0), gen_mel, mel_norm, sample_rate

    def perform_diffusion_from_codes_ar_prior(self, audio, sample_rate=22050):
        audio = audio.unsqueeze(0)

        mel = self.spec_fn({'in': audio})['out']
        mel_norm = normalize_mel(mel)
        cheater = self.local_modules['cheater_encoder'].to(audio.device)(mel_norm)
        cheater_codes = self.kmeans_inj({'in': cheater})['out']
        ar_latent = self.local_modules['ar_prior'].to(audio.device)(cheater_codes, cheater, return_latent=True)

        # 1. Generate the cheater latent using the input as a reference.
        sampler = self.diffuser.ddim_sample_loop if self.ddim else self.diffuser.p_sample_loop
        gen_cheater = sampler(self.model, cheater.shape, progress=True,
                              causal=self.causal, causal_slope=self.causal_slope,
                              model_kwargs={'codes': ar_latent})

        # 2. Decode the cheater into a MEL
        gen_mel = self.cheater_decoder_diffuser.ddim_sample_loop(self.local_modules['cheater_decoder'].diff.to(audio.device), (1,256,gen_cheater.shape[-1]*16), progress=True,
                                                 model_kwargs={'codes': gen_cheater.permute(0,2,1)})
        gen_mel_denorm = denormalize_mel(gen_mel)

        # 3. Decode into waveform.
        output_shape = (1,16,audio.shape[-1]//16)
        self.spec_decoder = self.spec_decoder.to(audio.device)
        gen_wav = self.spectral_diffuser.ddim_sample_loop(self.spec_decoder, output_shape, model_kwargs={'codes': gen_mel_denorm})
        gen_wav = pixel_shuffle_1d(gen_wav, 16)

        real_wav = self.spectral_diffuser.ddim_sample_loop(self.spec_decoder, output_shape, model_kwargs={'codes': mel})
        real_wav = pixel_shuffle_1d(real_wav, 16)

        return gen_wav, real_wav.squeeze(0), gen_mel, mel_norm, sample_rate

    def perform_fake_ar_reconstruction_from_cheater_gen(self, audio, sample_rate=22050):
        assert self.ddim, "DDIM mode expected for reconstructing cheater gen. Do you like to waste resources??"
        audio = audio.unsqueeze(0)

        mel = self.spec_fn({'in': audio})['out']
        mel_norm = normalize_mel(mel)
        cheater = self.local_modules['cheater_encoder'].to(audio.device)(mel_norm)

        # 1. Generate the cheater latent using the input as a reference.
        def diffuse(i, ref):
            mask = torch.zeros_like(ref)
            mask[:,:,:i] = 1
            return self.diffuser.p_sample_loop_with_guidance(self.model, ref, mask, model_kwargs={'conditioning_input': cheater})
        gen_cheater = torch.randn_like(cheater)
        for i in range(cheater.shape[-1]):
            gen_cheater = diffuse(i, gen_cheater)
            if i > 128:
                # abort early.
                gen_cheater = gen_cheater[:,:,:128]
                break

        # 2. Decode the cheater into a MEL. This operation and the next need to be chunked to make them feasible to perform within GPU memory.
        chunks = torch.split(gen_cheater, 64, dim=-1)
        gen_wavs = []
        for chunk in tqdm(chunks):
            gen_mel = self.cheater_decoder_diffuser.ddim_sample_loop(self.local_modules['cheater_decoder'].diff.to(audio.device), (1,256,chunk.shape[-1]*16), progress=True,
                                                     model_kwargs={'codes': chunk.permute(0,2,1)})

            # 3. And then the MEL back into a spectrogram
            output_shape = (1,16,audio.shape[-1]//(16*len(chunks)))
            self.spec_decoder = self.spec_decoder.to(audio.device)
            gen_mel_denorm = denormalize_mel(gen_mel)
            gen_wav = self.spectral_diffuser.p_sample_loop(self.spec_decoder, output_shape,
                                                  model_kwargs={'codes': gen_mel_denorm})
            gen_wav = pixel_shuffle_1d(gen_wav, 16)
            gen_wavs.append(gen_wav)
        gen_wav = torch.cat(gen_wavs, dim=-1)

        """ How to do progressive, causal decoding of the TFD diffuser:
        MAX_CONTEXT = 64
        def diffuse(start, len, guidance):
            mask = torch.zeros_like(guidance)
            mask[:,:,:(len-start)] = 1
            return self.cheater_decoder_diffuser.p_sample_loop_with_guidance(self.local_modules['cheater_decoder'].diff.to(audio.device),
                                                                             guidance_input=guidance, mask=mask,
                                                                             model_kwargs={'codes': gen_cheater[:,:,start:start+MAX_CONTEXT].permute(0,2,1)})
        guidance_mel = torch.zeros((1,256,MAX_CONTEXT*16), device=mel.device)
        gen_mel = torch.zeros((1,256,0), device=mel.device)
        for i in tqdm(list(range(gen_cheater.shape[-1]))):
            start = max(0, i-MAX_CONTEXT-1)
            l = min(16*(MAX_CONTEXT-1), i*16)
            ngm = diffuse(start, l, guidance_mel)
            gen_mel = torch.cat([gen_mel, ngm[:,:,l:l+16]], dim=-1)
            if gen_mel.shape[-1] < guidance_mel.shape[-1]:
                guidance_mel[:,:,:gen_mel.shape[-1]] = gen_mel
            else:
                guidance_mel = gen_mel[:,:,-guidance_mel.shape[-1]:]

        chunks = torch.split(gen_mel, MAX_CONTEXT*16, dim=-1)
        gen_wavs = []
        for chunk_mel in tqdm(chunks):
            # 3. And then the MEL back into a spectrogram
            output_shape = (1,16,audio.shape[-1]//(16*len(chunks)))
            self.spec_decoder = self.spec_decoder.to(audio.device)
            gen_mel_denorm = denormalize_mel(chunk_mel)
            gen_wav = self.spectral_diffuser.p_sample_loop(self.spec_decoder, output_shape,
                                                  model_kwargs={'codes': gen_mel_denorm})
            gen_wav = pixel_shuffle_1d(gen_wav, 16)
            gen_wavs.append(gen_wav)
        gen_wav = torch.cat(gen_wavs, dim=-1)
        """

        if audio.shape[-1] < 40 * 22050:
            real_wav = self.spectral_diffuser.p_sample_loop(self.spec_decoder, output_shape,
                                                  model_kwargs={'codes': mel})
            real_wav = pixel_shuffle_1d(real_wav, 16)
        else:
            real_wav = audio  # TODO: chunk like above.

        return gen_wav, real_wav.squeeze(0), gen_mel, mel_norm, sample_rate

    def project(self, sample, sample_rate):
        sample = torchaudio.functional.resample(sample, sample_rate, 22050)
        mel = self.spec_fn({'in': sample})['out']
        projection = self.projector.project(mel)
        return projection.squeeze(0)  # Getting rid of the batch dimension means it's just [hidden_dim]

    def compute_frechet_distance(self, proj1, proj2):
        # I really REALLY FUCKING HATE that this is going to numpy. Why does "pytorch_fid" operate in numpy land. WHY?
        proj1 = proj1.cpu().numpy()
        proj2 = proj2.cpu().numpy()
        mu1 = np.mean(proj1, axis=0)
        mu2 = np.mean(proj2, axis=0)
        sigma1 = np.cov(proj1, rowvar=False)
        sigma2 = np.cov(proj2, rowvar=False)
        try:
            return torch.tensor(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))
        except:
            return 0

    def perform_eval(self):
        save_path = osp.join(self.env['base_path'], "../", "audio_eval", str(self.env["step"]))
        os.makedirs(save_path, exist_ok=True)

        self.projector = self.projector.to(self.dev)
        self.projector.eval()

        # Attempt to fix the random state as much as possible. RNG state will be restored before returning.
        rng_state = torch.get_rng_state()
        torch.manual_seed(5)
        self.model.eval()

        with torch.no_grad():
            gen_projections = []
            real_projections = []
            for i in tqdm(list(range(0, len(self.data), self.skip))):
                path = self.data[(i + self.env['rank']) % len(self.data)]
                audio = load_audio(path, 22050).to(self.dev)
                #audio = load_audio('C:\\Users\\James\\Music\\another_longer_sample.wav', 22050).to(self.dev)  # <- hack, remove it!
                #audio = audio[:, :1764000]
                if self.clip:
                    audio = audio[:, :100000]
                sample, ref, sample_mel, ref_mel, sample_rate = self.diffusion_fn(audio)

                gen_projections.append(self.project(sample, sample_rate).cpu())  # Store on CPU to avoid wasting GPU memory.
                real_projections.append(self.project(ref, sample_rate).cpu())

                torchaudio.save(os.path.join(save_path, f"{self.env['rank']}_{i}_gen.wav"), sample.squeeze(0).cpu(), sample_rate)
                torchaudio.save(os.path.join(save_path, f"{self.env['rank']}_{i}_real.wav"), ref.cpu(), sample_rate)
                torchvision.utils.save_image((sample_mel.unsqueeze(1) + 1) / 2, os.path.join(save_path, f"{self.env['rank']}_{i}_gen_mel.png"))
                torchvision.utils.save_image((ref_mel.unsqueeze(1) + 1) / 2, os.path.join(save_path, f"{self.env['rank']}_{i}_real_mel.png"))
            gen_projections = torch.stack(gen_projections, dim=0)
            real_projections = torch.stack(real_projections, dim=0)
            frechet_distance = torch.tensor(self.compute_frechet_distance(gen_projections, real_projections), device=self.env['device'])

            if distributed.is_initialized() and distributed.get_world_size() > 1:
                distributed.all_reduce(frechet_distance)
                frechet_distance = frechet_distance / distributed.get_world_size()

        self.model.train()
        torch.set_rng_state(rng_state)

        # Put modules used for evaluation back into CPU memory.
        for k, mod in self.local_modules.items():
            self.local_modules[k] = mod.cpu()
        self.spec_decoder = self.spec_decoder.cpu()

        return {"frechet_distance": frechet_distance}


if __name__ == '__main__':
    diffusion = load_model_from_config('X:\\dlas\\experiments\\train_music_cheater_gen.yml', 'generator',
                                       also_load_savepoint=False,
                                       load_path='X:\\dlas\\experiments\\train_music_cheater_gen_v5_cosine_40_lyr\\models\\18500_generator_ema.pth'
                                       ).cuda()
    opt_eval = {'path': 'Y:\\split\\yt-music-eval',  # eval music, mostly electronica. :)
                #'path': 'E:\\music_eval',  # this is music from the training dataset, including a lot more variety.
                'diffusion_steps': 128,  # basis: 192
                'conditioning_free': False, 'conditioning_free_k': 1, 'use_ddim': True, 'clip_audio': False,
                'diffusion_schedule': 'cosine', 'diffusion_type': 'cheater_gen',
                # Slope 1: 1.03x, 2: 1.06, 4: 1.135, 8: 1.27, 16: 1.54
                #'causal': True, 'causal_slope': 4,  # DONT FORGET TO INCREMENT THE STEP!
                #'partial_low': 128, 'partial_high': 192
    }
    env = {'rank': 0, 'base_path': 'D:\\tmp\\test_eval_music', 'step': 201, 'device': 'cuda', 'opt': {}}
    eval = MusicDiffusionFid(diffusion, opt_eval, env)
    fds = []
    for i in range(2):
        res = eval.perform_eval()
        print(res)
        fds.append(res['frechet_distance'])
    print(fds)

