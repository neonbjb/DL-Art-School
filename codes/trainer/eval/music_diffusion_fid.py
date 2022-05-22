import os
import os.path as osp
from glob import glob
from random import shuffle

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
    normalize_mel
from utils.music_utils import get_music_codegen, get_mel2wav_model
from utils.util import opt_get, load_model_from_config


class MusicDiffusionFid(evaluator.Evaluator):
    """
    Evaluator produces generate from a music diffusion model.
    """
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env, uses_all_ddp=True)
        self.real_path = opt_eval['path']
        self.data = self.load_data(self.real_path)
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
        self.dev = self.env['device']
        mode = opt_get(opt_eval, ['diffusion_type'], 'tts')

        self.spec_decoder = get_mel2wav_model()
        self.projector = ContrastiveAudio(model_dim=512, transformer_heads=8, dropout=0, encoder_depth=8, mel_channels=256)
        self.projector.load_state_dict(torch.load('../experiments/music_eval_projector.pth', map_location=torch.device('cpu')))
        self.local_modules = {'spec_decoder': self.spec_decoder, 'projector': self.projector}

        if mode == 'spec_decode':
            self.diffusion_fn = self.perform_diffusion_spec_decode
        elif 'gap_fill_' in mode:
            self.diffusion_fn = self.perform_diffusion_gap_fill
            if '_freq' in mode:
                self.gap_gen_fn = self.gen_freq_gap
            else:
                self.gap_gen_fn = self.gen_time_gap
        elif 'rerender' in mode:
            self.diffusion_fn = self.perform_rerender
        elif 'from_codes' == mode:
            self.diffusion_fn = self.perform_diffusion_from_codes
            self.local_modules['codegen'] = get_music_codegen()
        self.spec_fn = TorchMelSpectrogramInjector({'n_mel_channels': 256, 'mel_fmax': 22000, 'normalize': True, 'in': 'in', 'out': 'out'}, {})

    def load_data(self, path):
        return list(glob(f'{path}/*.wav'))

    def perform_diffusion_spec_decode(self, audio, sample_rate=22050):
        if sample_rate != sample_rate:
            real_resampled = torchaudio.functional.resample(audio, 22050, sample_rate).unsqueeze(0)
        else:
            real_resampled = audio
        audio = audio.unsqueeze(0)
        output_shape = (1, 16, audio.shape[-1] // 16)
        mel = self.spec_fn({'in': audio})['out']
        gen = self.diffuser.p_sample_loop(self.model, output_shape, noise=torch.zeros(*output_shape, device=audio.device),
                                          model_kwargs={'aligned_conditioning': mel})
        gen = pixel_shuffle_1d(gen, 16)

        return gen, real_resampled, normalize_mel(self.spec_fn({'in': gen})['out']), normalize_mel(mel), sample_rate

    def gen_freq_gap(self, mel, band_range=(60,100)):
        gap_start, gap_end = band_range
        mask = torch.ones_like(mel)
        mask[:, gap_start:gap_end] = 0
        return mel * mask, mask

    def gen_time_gap(self, mel):
        mask = torch.ones_like(mel)
        mask[:, :, 86*4:86*6] = 0
        return mel * mask, mask

    def perform_diffusion_gap_fill(self, audio, sample_rate=22050):
        if sample_rate != sample_rate:
            real_resampled = torchaudio.functional.resample(audio, 22050, sample_rate).unsqueeze(0)
        else:
            real_resampled = audio
        audio = audio.unsqueeze(0)

        # Fetch the MEL and mask out the requested bands.
        mel = self.spec_fn({'in': audio})['out']
        mel = normalize_mel(mel)
        mel, mask = self.gap_gen_fn(mel)

        # Repair the MEL with the given model.
        spec = self.diffuser.p_sample_loop_with_guidance(self.model, mel, mask, model_kwargs={'truth': mel})
        spec = denormalize_mel(spec)

        # Re-convert the resulting MEL back into audio using the spectrogram decoder.
        output_shape = (1, 16, audio.shape[-1] // 16)
        self.spec_decoder = self.spec_decoder.to(audio.device)
        # Cool fact: we can re-use the diffuser for the spectrogram diffuser since it has the same parametrization.
        gen = self.diffuser.p_sample_loop(self.spec_decoder, output_shape, noise=torch.zeros(*output_shape, device=audio.device),
                                          model_kwargs={'aligned_conditioning': spec})
        gen = pixel_shuffle_1d(gen, 16)

        return gen, real_resampled, normalize_mel(spec), mel, sample_rate

    def perform_rerender(self, audio, sample_rate=22050):
        if sample_rate != sample_rate:
            real_resampled = torchaudio.functional.resample(audio, 22050, sample_rate).unsqueeze(0)
        else:
            real_resampled = audio
        audio = audio.unsqueeze(0)

        # Fetch the MEL and mask out the requested bands.
        mel = self.spec_fn({'in': audio})['out']
        mel = normalize_mel(mel)

        segments = [(0,10),(10,25),(25,45),(45,60),(60,80),(80,100),(100,130),(130,170),(170,210),(210,256)]
        shuffle(segments)
        spec = mel
        for i, segment in enumerate(segments):
            mel, mask = self.gen_freq_gap(mel, band_range=segment)
            # Repair the MEL with the given model.
            spec = self.diffuser.p_sample_loop_with_guidance(self.model, spec, mask, model_kwargs={'truth': spec})
            torchvision.utils.save_image((spec.unsqueeze(1) + 1) / 2, f"{i}_rerender.png")

        spec = denormalize_mel(spec)

        # Re-convert the resulting MEL back into audio using the spectrogram decoder.
        output_shape = (1, 16, audio.shape[-1] // 16)
        self.spec_decoder = self.spec_decoder.to(audio.device)
        # Cool fact: we can re-use the diffuser for the spectrogram diffuser since it has the same parametrization.
        gen = self.diffuser.p_sample_loop(self.spec_decoder, output_shape, noise=torch.zeros(*output_shape, device=audio.device),
                                          model_kwargs={'aligned_conditioning': spec})
        gen = pixel_shuffle_1d(gen, 16)

        return gen, real_resampled, normalize_mel(spec), mel, sample_rate

    def perform_diffusion_from_codes(self, audio, sample_rate=22050):
        if sample_rate != sample_rate:
            real_resampled = torchaudio.functional.resample(audio, 22050, sample_rate).unsqueeze(0)
        else:
            real_resampled = audio
        audio = audio.unsqueeze(0)

        # Fetch the MEL and mask out the requested bands.
        mel = self.spec_fn({'in': audio})['out']
        codegen = self.local_modules['codegen'].to(mel.device)
        codes = codegen.get_codes(mel)
        mel_norm = normalize_mel(mel)
        precomputed_codes, precomputed_cond = self.model.timestep_independent(codes=codes, conditioning_input=mel_norm[:,:,:112],
                                                      expected_seq_len=mel_norm.shape[-1], return_code_pred=False)
        gen_mel = self.diffuser.p_sample_loop(self.model, mel_norm.shape, noise=torch.zeros_like(mel_norm),
                                              model_kwargs={'precomputed_code_embeddings': precomputed_codes, 'precomputed_cond_embeddings': precomputed_cond})

        gen_mel_denorm = denormalize_mel(gen_mel)
        output_shape = (1,16,audio.shape[-1]//16)
        self.spec_decoder = self.spec_decoder.to(audio.device)
        gen_wav = self.diffuser.p_sample_loop(self.spec_decoder, output_shape, model_kwargs={'aligned_conditioning': gen_mel_denorm})
        gen_wav = pixel_shuffle_1d(gen_wav, 16)

        return gen_wav, real_resampled, gen_mel, mel_norm, sample_rate


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
        return torch.tensor(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))

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
                path = self.data[i + self.env['rank']]
                audio = load_audio(path, 22050).to(self.dev)
                audio = audio[:, :22050*10]
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

        return {"frechet_distance": frechet_distance}


if __name__ == '__main__':
    diffusion = load_model_from_config('X:\\dlas\\experiments\\train_music_diffusion_flat.yml', 'generator',
                                       also_load_savepoint=False,
                                       load_path='X:\\dlas\\experiments\\train_music_diffusion_flat\\models\\33000_generator_ema.pth').cuda()
    opt_eval = {'path': 'Y:\\split\\yt-music-eval', 'diffusion_steps': 100,
                'conditioning_free': False, 'conditioning_free_k': 1,
                'diffusion_schedule': 'linear', 'diffusion_type': 'from_codes'}
    env = {'rank': 0, 'base_path': 'D:\\tmp\\test_eval_music', 'step': 25, 'device': 'cuda', 'opt': {}}
    eval = MusicDiffusionFid(diffusion, opt_eval, env)
    print(eval.perform_eval())
