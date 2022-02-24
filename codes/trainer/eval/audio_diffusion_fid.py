import os
import os.path as osp
import torch
import torchaudio
import torchvision
from pytorch_fid import fid_score
from pytorch_fid.fid_score import calculate_frechet_distance
from torch import distributed
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch.nn.functional as F
import numpy as np

import trainer.eval.evaluator as evaluator
from data.audio.paired_voice_audio_dataset import load_tsv_aligned_codes
from data.audio.unsupervised_audio_dataset import load_audio
from models.clip.mel_text_clip import MelTextCLIP
from models.tacotron2.text import sequence_to_text, text_to_sequence
from scripts.audio.gen.speech_synthesis_utils import load_discrete_vocoder_diffuser, wav_to_mel, load_speech_dvae, \
    convert_mel_to_codes
from utils.util import ceil_multiple, opt_get


class AudioDiffusionFid(evaluator.Evaluator):
    """
    Evaluator produces generate from a diffusion model, then uses a CLIP model to judge the similarity between text & speech.
    """
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env, uses_all_ddp=True)
        self.real_path = opt_eval['eval_tsv']
        self.data = load_tsv_aligned_codes(self.real_path)
        if distributed.is_initialized() and distributed.get_world_size() > 1:
            self.skip = distributed.get_world_size()  # One batch element per GPU.
        else:
            self.skip = 1
        diffusion_steps = opt_get(opt_eval, ['diffusion_steps'], 50)
        diffusion_schedule = opt_get(env['opt'], ['steps', 'generator', 'injectors', 'diffusion', 'beta_schedule', 'schedule_name'], None)
        if diffusion_schedule is None:
            print("Unable to infer diffusion schedule from master options. Getting it from eval (or guessing).")
            diffusion_schedule = opt_get(opt_eval, ['diffusion_schedule'], 'cosine')
        self.diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=diffusion_steps, schedule=diffusion_schedule)
        self.dev = self.env['device']
        mode = opt_get(opt_eval, ['diffusion_type'], 'tts')
        if mode == 'tts':
            self.diffusion_fn = self.perform_diffusion_tts
        elif mode == 'vocoder':
            self.dvae = load_speech_dvae()
            self.dvae.eval()
            self.diffusion_fn = self.perform_diffusion_vocoder

    def perform_diffusion_tts(self, audio, codes, text, sample_rate=5500):
        real_resampled = torchaudio.functional.resample(audio, 22050, sample_rate).unsqueeze(0)
        aligned_codes_compression_factor = sample_rate * 221 // 11025
        output_size = codes.shape[-1]*aligned_codes_compression_factor
        padded_size = ceil_multiple(output_size, 2048)
        padding_added = padded_size - output_size
        padding_needed_for_codes = padding_added // aligned_codes_compression_factor
        if padding_needed_for_codes > 0:
            codes = F.pad(codes, (0, padding_needed_for_codes))
        output_shape = (1, 1, padded_size)
        gen = self.diffuser.p_sample_loop(self.model, output_shape,
                                    model_kwargs={'tokens': codes.unsqueeze(0),
                                                  'conditioning_input': real_resampled})
        return gen, real_resampled, sample_rate

    def perform_diffusion_vocoder(self, audio, codes, text, sample_rate=5500):
        mel = wav_to_mel(audio)
        mel_codes = convert_mel_to_codes(self.dvae, mel)
        text_codes = text_to_sequence(text)
        real_resampled = torchaudio.functional.resample(audio, 22050, sample_rate).unsqueeze(0)

        output_size = real_resampled.shape[-1]
        aligned_codes_compression_factor = output_size // mel_codes.shape[-1]
        padded_size = ceil_multiple(output_size, 2048)
        padding_added = padded_size - output_size
        padding_needed_for_codes = padding_added // aligned_codes_compression_factor
        if padding_needed_for_codes > 0:
            mel_codes = F.pad(mel_codes, (0, padding_needed_for_codes))
        output_shape = (1, 1, padded_size)
        gen = self.diffuser.p_sample_loop(self.model, output_shape,
                                    model_kwargs={'tokens': mel_codes,
                                                  'conditioning_input': audio.unsqueeze(0),
                                                  'unaligned_input': torch.tensor(text_codes, device=audio.device).unsqueeze(0)})
        return gen, real_resampled, sample_rate

    def load_projector(self):
        """
        Builds the CLIP model used to project speech into a latent. This model has fixed parameters and a fixed loading
        path for the time being.
        """
        model = MelTextCLIP(dim_text=512, dim_latent=512, dim_speech=512, num_text_tokens=148, text_enc_depth=8,
                            text_seq_len=400, text_heads=8, speech_enc_depth=10, speech_heads=8, speech_seq_len=1000,
                            text_mask_percentage=.15, voice_mask_percentage=.15)
        weights = torch.load('../experiments/clip_text_to_voice_for_speech_fid.pth')
        model.load_state_dict(weights)
        return model

    def project(self, projector, sample, sample_rate):
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

        projector = self.load_projector().to(self.env['device'])
        projector.eval()
        if hasattr(self, 'dvae'):
            self.dvae = self.dvae.to(self.env['device'])

        # Attempt to fix the random state as much as possible. RNG state will be restored before returning.
        rng_state = torch.get_rng_state()
        torch.manual_seed(5)
        self.model.eval()

        with torch.no_grad():
            gen_projections = []
            real_projections = []
            for i in tqdm(list(range(0, len(self.data), self.skip))):
                path, text, codes = self.data[i + self.env['rank']]
                audio = load_audio(path, 22050).to(self.dev)
                codes = codes.to(self.dev)
                sample, ref, sample_rate = self.diffusion_fn(audio, codes, text)

                gen_projections.append(self.project(projector, sample, sample_rate).cpu())  # Store on CPU to avoid wasting GPU memory.
                real_projections.append(self.project(projector, ref, sample_rate).cpu())

                torchaudio.save(os.path.join(save_path, f"{self.env['rank']}_{i}_gen.wav"), sample.squeeze(0).cpu(), sample_rate)
                torchaudio.save(os.path.join(save_path, f"{self.env['rank']}_{i}_real.wav"), ref.squeeze(0).cpu(), sample_rate)
            gen_projections = torch.stack(gen_projections, dim=0)
            real_projections = torch.stack(real_projections, dim=0)
            frechet_distance = torch.tensor(self.compute_frechet_distance(gen_projections, real_projections), device=self.env['device'])

            if distributed.is_initialized() and distributed.get_world_size() > 1:
                distributed.all_reduce(frechet_distance)
                frechet_distance = frechet_distance / distributed.get_world_size()

        self.model.train()
        if hasattr(self, 'dvae'):
            self.dvae = self.dvae.to('cpu')
        torch.set_rng_state(rng_state)

        return {"frechet_distance": frechet_distance}


if __name__ == '__main__':
    from utils.util import load_model_from_config

    diffusion = load_model_from_config('X:\\dlas\\experiments\\train_diffusion_tts7_dvae_thin_with_text.yml', 'generator',
                                       also_load_savepoint=False, load_path='X:\\dlas\\experiments\\train_diffusion_tts7_dvae_thin_with_text\\models\\5500_generator_ema.pth').cuda()
    opt_eval = {'eval_tsv': 'Y:\\libritts\\test-clean\\transcribed-brief-w2v.tsv', 'diffusion_steps': 50,
                'diffusion_schedule': 'linear', 'diffusion_type': 'vocoder'}
    env = {'rank': 0, 'base_path': 'D:\\tmp\\test_eval', 'step': 500, 'device': 'cuda', 'opt': {}}
    eval = AudioDiffusionFid(diffusion, opt_eval, env)
    print(eval.perform_eval())