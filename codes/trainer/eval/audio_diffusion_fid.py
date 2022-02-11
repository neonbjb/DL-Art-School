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
from scripts.audio.gen.speech_synthesis_utils import load_discrete_vocoder_diffuser
from utils.util import ceil_multiple, opt_get


class AudioDiffusionFid(evaluator.Evaluator):
    """
    Evaluator produces generate from a diffusion model, then uses a pretrained wav2vec model to compute a frechet
    distance between real and fake samples.
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

    def perform_diffusion(self, audio, codes, sample_rate=5500):
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

    def project(self, projector, sample, sample_rate):
        sample = torchaudio.functional.resample(sample, sample_rate, 16000)
        sample = (sample - sample.mean()) / torch.sqrt(sample.var() + 1e-7)
        return projector(sample.squeeze(1), output_hidden_states=True).hidden_states[-1].squeeze(0)  # Getting rid of the batch dimension means it's just [seq_len,hidden_states]

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

        projector = Wav2Vec2ForCTC.from_pretrained(f"facebook/wav2vec2-large").to(self.dev)
        projector.eval()

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
                sample, ref, sample_rate = self.perform_diffusion(audio, codes)

                gen_projections.append(self.project(projector, sample, sample_rate).cpu())  # Store on CPU to avoid wasting GPU memory.
                real_projections.append(self.project(projector, ref, sample_rate).cpu())

                torchaudio.save(os.path.join(save_path, f"{self.env['rank']}_{i}_gen.wav"), sample.squeeze(0).cpu(), sample_rate)
                torchaudio.save(os.path.join(save_path, f"{self.env['rank']}_{i}_real.wav"), ref.squeeze(0).cpu(), sample_rate)
            gen_projections = torch.cat(gen_projections, dim=0)
            real_projections = torch.cat(real_projections, dim=0)
            fid = self.compute_frechet_distance(gen_projections, real_projections)

            if distributed.is_initialized() and distributed.get_world_size() > 1:
                fid = distributed.all_reduce(fid) / distributed.get_world_size()

        self.model.train()
        torch.set_rng_state(rng_state)

        return {"fid": fid}


if __name__ == '__main__':
    from utils.util import load_model_from_config

    diffusion = load_model_from_config('X:\\dlas\\experiments\\train_diffusion_tts5_medium.yml', 'generator',
                                       also_load_savepoint=False, load_path='X:\\dlas\\experiments\\train_diffusion_tts5_medium\\models\\73000_generator_ema.pth').cuda()
    opt_eval = {'eval_tsv': 'Y:\\libritts\\test-clean\\transcribed-brief-w2v.tsv', 'diffusion_steps': 50}
    env = {'rank': 0, 'base_path': 'D:\\tmp\\test_eval', 'step': 500, 'device': 'cuda'}
    eval = AudioDiffusionFid(diffusion, opt_eval, env)
    eval.perform_eval()