import os
import os.path as osp
from glob import glob

import torch
import torchaudio
import torchvision
from pytorch_fid.fid_score import calculate_frechet_distance
from torch import distributed
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC
import torch.nn.functional as F
import numpy as np

import trainer.eval.evaluator as evaluator
from data.audio.paired_voice_audio_dataset import load_tsv_aligned_codes
from data.audio.unsupervised_audio_dataset import load_audio
from data.audio.voice_tokenizer import VoiceBpeTokenizer
from models.audio.music.unet_diffusion_waveform_gen import DiffusionWaveformGen
from models.clip.mel_text_clip import MelTextCLIP
from models.audio.tts.tacotron2 import text_to_sequence
from models.diffusion.gaussian_diffusion import get_named_beta_schedule
from models.diffusion.respace import space_timesteps, SpacedDiffusion
from scripts.audio.gen.speech_synthesis_utils import load_discrete_vocoder_diffuser, wav_to_mel, load_speech_dvae, \
    convert_mel_to_codes, load_univnet_vocoder, wav_to_univnet_mel
from trainer.injectors.audio_injectors import denormalize_mel, TorchMelSpectrogramInjector, pixel_shuffle_1d, \
    normalize_mel
from utils.util import ceil_multiple, opt_get, load_model_from_config, pad_or_truncate


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

        self.spec_decoder = DiffusionWaveformGen(model_channels=256, in_channels=16, in_mel_channels=256, out_channels=32,
                                                 channel_mult=[1,2,3,4], num_res_blocks=[3,3,3,3], token_conditioning_resolutions=[1,4],
                                                 num_heads=8,
                                                 dropout=0, kernel_size=3, scale_factor=2, time_embed_dim_multiplier=4, unconditioned_percentage=0)
        self.spec_decoder.load_state_dict(torch.load('../experiments/music_waveform_gen.pth', map_location=torch.device('cpu')))
        self.local_modules = {'spec_decoder': self.spec_decoder}

        if mode == 'spec_decode':
            self.diffusion_fn = self.perform_diffusion_spec_decode
        elif 'gap_fill_' in mode:
            self.diffusion_fn = self.perform_diffusion_gap_fill
            if '_freq' in mode:
                self.gap_gen_fn = self.gen_freq_gap
            else:
                self.gap_gen_fn = self.gen_time_gap
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

        return gen, real_resampled, sample_rate

    def gen_freq_gap(self, mel, band_range=(130,150)):
        gap_start, gap_end = band_range
        mel[:, gap_start:gap_end] = 0
        return mel

    def gen_time_gap(self, mel):
        mel[:, :, 22050*5:22050*6] = 0
        return mel

    def perform_diffusion_gap_fill(self, audio, sample_rate=22050, band_range=(130,150)):
        if sample_rate != sample_rate:
            real_resampled = torchaudio.functional.resample(audio, 22050, sample_rate).unsqueeze(0)
        else:
            real_resampled = audio
        audio = audio.unsqueeze(0)

        # Fetch the MEL and mask out the requested bands.
        mel = self.spec_fn({'in': audio})['out']
        mel = normalize_mel(mel)
        mel = self.gap_gen_fn(mel)
        output_shape = (1, mel.shape[1], mel.shape[2])

        # Repair the MEL with the given model.
        spec = self.diffuser.p_sample_loop(self.model, output_shape, noise=torch.zeros(*output_shape, device=audio.device),
                                          model_kwargs={'truth': mel})
        import torchvision
        torchvision.utils.save_image((spec.unsqueeze(1) + 1) / 2, 'gen.png')
        torchvision.utils.save_image((mel.unsqueeze(1) + 1) / 2, 'mel.png')
        spec = denormalize_mel(spec)

        # Re-convert the resulting MEL back into audio using the spectrogram decoder.
        output_shape = (1, 16, audio.shape[-1] // 16)
        self.spec_decoder = self.spec_decoder.to(audio.device)
        # Cool fact: we can re-use the diffuser for the spectrogram diffuser since it has the same parametrization.
        gen = self.diffuser.p_sample_loop(self.spec_decoder, output_shape, noise=torch.zeros(*output_shape, device=audio.device),
                                          model_kwargs={'aligned_conditioning': spec})
        gen = pixel_shuffle_1d(gen, 16)

        return gen, real_resampled, sample_rate

    def load_projector(self):
        # TODO: implement for music.
        model = MelTextCLIP(dim_text=512, dim_latent=512, dim_speech=512, num_text_tokens=148, text_enc_depth=8,
                            text_seq_len=400, text_heads=8, speech_enc_depth=10, speech_heads=8, speech_seq_len=1000,
                            text_mask_percentage=.15, voice_mask_percentage=.15)
        weights = torch.load('../experiments/clip_text_to_voice_for_speech_fid.pth')
        model.load_state_dict(weights)
        return model

    def project(self, projector, sample, sample_rate):
        # TODO: implement for music.
        sample = torchaudio.functional.resample(sample, sample_rate, 22050)
        mel = wav_to_mel(sample)
        return projector.get_speech_projection(mel).squeeze(0)  # Getting rid of the batch dimension means it's just [hidden_dim]

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

        #projector = self.load_projector().to(self.env['device'])
        #projector.eval()

        # Attempt to fix the random state as much as possible. RNG state will be restored before returning.
        rng_state = torch.get_rng_state()
        torch.manual_seed(5)
        self.model.eval()

        frechet_distance = 0
        with torch.no_grad():
            gen_projections = []
            real_projections = []
            for i in tqdm(list(range(0, len(self.data), self.skip))):
                path = self.data[i + self.env['rank']]
                audio = load_audio(path, 22050).to(self.dev)
                mel = self.spec_fn({'in': audio})['out']
                mel_norm = (mel + mel.min().abs())
                mel_norm = mel_norm / mel_norm.max(dim=-1, keepdim=True).values
                torchvision.utils.save_image(mel_norm.unsqueeze(1), 'mel.png')
                sample, ref, sample_rate = self.diffusion_fn(audio)

                #gen_projections.append(self.project(projector, sample, sample_rate).cpu())  # Store on CPU to avoid wasting GPU memory.
                #real_projections.append(self.project(projector, ref, sample_rate).cpu())

                torchaudio.save(os.path.join(save_path, f"{self.env['rank']}_{i}_gen.wav"), sample.squeeze(0).cpu(), sample_rate)
                torchaudio.save(os.path.join(save_path, f"{self.env['rank']}_{i}_real.wav"), ref.cpu(), sample_rate)
            #gen_projections = torch.stack(gen_projections, dim=0)
            #real_projections = torch.stack(real_projections, dim=0)
            #frechet_distance = torch.tensor(self.compute_frechet_distance(gen_projections, real_projections), device=self.env['device'])

            #if distributed.is_initialized() and distributed.get_world_size() > 1:
            #    distributed.all_reduce(frechet_distance)
            #    frechet_distance = frechet_distance / distributed.get_world_size()
            #    distributed.all_reduce(intelligibility_loss)
            #    intelligibility_loss = intelligibility_loss / distributed.get_world_size()

        self.model.train()
        torch.set_rng_state(rng_state)

        # Put modules used for evaluation back into CPU memory.
        for k, mod in self.local_modules.items():
            self.local_modules[k] = mod.cpu()

        return {"frechet_distance": frechet_distance}


if __name__ == '__main__':
    diffusion = load_model_from_config('X:\\dlas\\experiments\\train_music_gap_filler.yml', 'generator',
                                       also_load_savepoint=False,
                                       load_path='X:\\dlas\\experiments\\train_music_gap_filler\\models\\5000_generator.pth').cuda()
    opt_eval = {'path': 'Y:\\split\\yt-music-eval', 'diffusion_steps': 100,
                'conditioning_free': False, 'conditioning_free_k': 1,
                'diffusion_schedule': 'linear', 'diffusion_type': 'gap_fill_freq'}
    env = {'rank': 0, 'base_path': 'D:\\tmp\\test_eval_music', 'step': 2, 'device': 'cuda', 'opt': {}}
    eval = MusicDiffusionFid(diffusion, opt_eval, env)
    print(eval.perform_eval())
