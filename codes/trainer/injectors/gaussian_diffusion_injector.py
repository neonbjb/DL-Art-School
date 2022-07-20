import functools
import random

import torch
from torch.cuda.amp import autocast

from models.diffusion.gaussian_diffusion import get_named_beta_schedule
from models.diffusion.resample import create_named_schedule_sampler, LossAwareSampler, DeterministicSampler, LossSecondMomentResampler
from models.diffusion.respace import space_timesteps, SpacedDiffusion
from trainer.inject import Injector
from utils.util import opt_get


def masked_channel_balancer(inp, proportion=1):
    with torch.no_grad():
        only_channels = inp.mean(dim=(0,2))  # Only currently works for audio tensors. Could be retrofitted for 2d (or 3D!) modalities.
        dist = only_channels / only_channels.sum()
        dist_mult = only_channels.shape[0] * proportion
        dist = (dist * dist_mult).clamp(0, 1)
        mask = torch.bernoulli(dist)
    return inp * mask.view(1,inp.shape[1],1)


def channel_restriction(inp, low, high):
    assert low > 0 and low < inp.shape[1] and high <= inp.shape[1]
    m = torch.zeros_like(inp)
    m[:,low:high] = 1
    return inp * m


# Injects a gaussian diffusion loss as described by OpenAIs "Improved Denoising Diffusion Probabilistic Models" paper.
# Largely uses OpenAI's own code to do so (all code from models.diffusion.*)
class GaussianDiffusionInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.generator = opt['generator']
        self.output_variational_bounds_key = opt['out_key_vb_loss']
        self.output_x_start_key = opt['out_key_x_start']
        opt['diffusion_args']['betas'] = get_named_beta_schedule(**opt['beta_schedule'])
        opt['diffusion_args']['use_timesteps'] = space_timesteps(opt['beta_schedule']['num_diffusion_timesteps'],
                                                                 [opt['beta_schedule']['num_diffusion_timesteps']])
        self.diffusion = SpacedDiffusion(**opt['diffusion_args'])
        self.schedule_sampler = create_named_schedule_sampler(opt['sampler_type'], self.diffusion)
        self.model_input_keys = opt_get(opt, ['model_input_keys'], [])
        self.extra_model_output_keys = opt_get(opt, ['extra_model_output_keys'], [])
        self.deterministic_timesteps_every = opt_get(opt, ['deterministic_timesteps_every'], 0)
        self.deterministic_sampler = DeterministicSampler(self.diffusion, opt_get(opt, ['deterministic_sampler_expected_batch_size'], 2048), env)
        self.causal_mode = opt_get(opt, ['causal_mode'], False)
        self.causal_slope_range = opt_get(opt, ['causal_slope_range'], [1,8])
        self.preprocess_fn = opt_get(opt, ['preprocess_fn'], None)

        k = 0
        if 'channel_balancer_proportion' in opt.keys():
            self.channel_balancing_fn = functools.partial(masked_channel_balancer, proportion=opt['channel_balancer_proportion'])
            k += 1
        if 'channel_restriction_low' in opt.keys():
            self.channel_balancing_fn = functools.partial(channel_restriction, low=opt['channel_restriction_low'], high=opt['channel_restriction_high'])
            k += 1
        if not hasattr(self, 'channel_balancing_fn'):
            self.channel_balancing_fn = None
        assert k <= 1, 'Only one channel filtering function can be applied.'

        self.num_timesteps = opt['beta_schedule']['num_diffusion_timesteps']
        self.latest_mse_by_batch = torch.tensor([0])
        self.latest_timesteps = torch.tensor([0])

    def extra_metrics(self):
        uqt = self.latest_timesteps > self.num_timesteps * 3 / 4
        uql = (self.latest_mse_by_batch * uqt).sum() / uqt.sum() if uqt.sum() != 0 else 0
        muqt = (self.latest_timesteps > self.num_timesteps / 2) * (self.latest_timesteps < self.num_timesteps * 3 / 4)
        muql = (self.latest_mse_by_batch * muqt).sum() / muqt.sum() if muqt.sum() != 0 else 0
        d = {
            'upper_quantile_mse_loss': uql,
            'mid_upper_quantile_mse_loss': muql,
        }
        if hasattr(self, 'schedule_sampler') and isinstance(self.schedule_sampler, LossSecondMomentResampler):
            d['sampler_warmed_up'] = torch.tensor(float(self.schedule_sampler._warmed_up()))
        return d

    def forward(self, state):
        gen = self.env['generators'][self.opt['generator']]
        hq = state[self.input]
        assert hq.max() < 1.000001 or hq.min() > -1.00001, f"Attempting to train gaussian diffusion on un-normalized inputs. This won't work, silly! {hq.min()} {hq.max()}"

        with autocast(enabled=self.env['opt']['fp16']):
            if not gen.training or (self.deterministic_timesteps_every != 0 and self.env['step'] % self.deterministic_timesteps_every == 0):
                sampler = self.deterministic_sampler
            else:
                sampler = self.schedule_sampler
                self.deterministic_sampler.reset()  # Keep this reset whenever it is not being used, so it is ready to use automatically.
            model_inputs = {k: state[v] if isinstance(v, str) else v for k, v in self.model_input_keys.items()}
            t, weights = sampler.sample(hq.shape[0], hq.device)

            if self.preprocess_fn is not None:
                hq = getattr(gen.module, self.preprocess_fn)(hq, t, self.diffusion)
            if self.causal_mode:
                cs, ce = self.causal_slope_range
                slope = random.random() * (ce-cs) + cs
                diffusion_outputs = self.diffusion.causal_training_losses(gen, hq, t, model_kwargs=model_inputs,
                                                                   channel_balancing_fn=self.channel_balancing_fn,
                                                                          causal_slope=slope)
            else:
                diffusion_outputs = self.diffusion.training_losses(gen, hq, t, model_kwargs=model_inputs,
                                                                   channel_balancing_fn=self.channel_balancing_fn)

            if isinstance(sampler, LossAwareSampler):
                sampler.update_with_local_losses(t, diffusion_outputs['loss'])
            if len(self.extra_model_output_keys) > 0:
                assert(len(self.extra_model_output_keys) == len(diffusion_outputs['extra_outputs']))
                out = {k: v for k, v in zip(self.extra_model_output_keys, diffusion_outputs['extra_outputs'])}
            else:
                out = {}
            out.update({self.output: diffusion_outputs['mse'],
                    self.output_variational_bounds_key: diffusion_outputs['vb'],
                    self.output_x_start_key: diffusion_outputs['x_start_predicted']})
            self.latest_mse_by_batch = diffusion_outputs['mse_by_batch'].detach().clone()
            self.latest_timesteps = t.clone()

        return out


def closest_multiple(inp, multiple):
    div = inp // multiple
    mod = inp % multiple
    if mod == 0:
        return inp
    else:
        return int((div+1)*multiple)


# Performs inference using a network trained to predict a reverse diffusion process, which nets a image.
class GaussianDiffusionInferenceInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        use_ddim = opt_get(opt, ['use_ddim'], False)
        self.generator = opt['generator']
        self.output_batch_size = opt['output_batch_size']
        self.output_scale_factor = opt['output_scale_factor']
        self.undo_n1_to_1 = opt_get(opt, ['undo_n1_to_1'], False)  # Explanation: when specified, will shift the output of this injector from [-1,1] to [0,1]
        opt['diffusion_args']['betas'] = get_named_beta_schedule(**opt['beta_schedule'])
        if use_ddim:
            spacing = "ddim" + str(opt['respaced_timestep_spacing'])
        else:
            spacing = [opt_get(opt, ['respaced_timestep_spacing'], opt['beta_schedule']['num_diffusion_timesteps'])]
        opt['diffusion_args']['use_timesteps'] = space_timesteps(opt['beta_schedule']['num_diffusion_timesteps'], spacing)
        self.diffusion = SpacedDiffusion(**opt['diffusion_args'])
        self.sampling_fn = self.diffusion.ddim_sample_loop if use_ddim else self.diffusion.p_sample_loop
        self.model_input_keys = opt_get(opt, ['model_input_keys'], [])
        self.use_ema_model = opt_get(opt, ['use_ema'], False)
        self.noise_style = opt_get(opt, ['noise_type'], 'random')  # 'zero', 'fixed' or 'random'
        self.multiple_requirement = opt_get(opt, ['multiple_requirement'], 4096)

    def forward(self, state):
        if self.use_ema_model:
            gen = self.env['emas'][self.opt['generator']]
        else:
            gen = self.env['generators'][self.opt['generator']]
        model_inputs = {k: state[v][:self.output_batch_size] for k, v in self.model_input_keys.items()}
        gen.eval()
        with torch.no_grad():
            if 'low_res' in model_inputs.keys():
                output_shape = (self.output_batch_size, 3, model_inputs['low_res'].shape[-2] * self.output_scale_factor,
                                model_inputs['low_res'].shape[-1] * self.output_scale_factor)
                dev = model_inputs['low_res'].device
            elif 'spectrogram' in model_inputs.keys():
                output_shape = (self.output_batch_size, 1, closest_multiple(model_inputs['spectrogram'].shape[-1] * self.output_scale_factor, self.multiple_requirement))
                dev = model_inputs['spectrogram'].device
            elif 'discrete_spectrogram' in model_inputs.keys():
                output_shape = (self.output_batch_size, 1, closest_multiple(model_inputs['discrete_spectrogram'].shape[-1]*1024, self.multiple_requirement))
                dev = model_inputs['discrete_spectrogram'].device
            else:
                raise NotImplementedError
            noise = None
            if self.noise_style == 'zero':
                noise = torch.zeros(output_shape, device=dev)
            elif self.noise_style == 'fixed':
                if not hasattr(self, 'fixed_noise') or self.fixed_noise.shape != output_shape:
                    self.fixed_noise = torch.randn(output_shape, device=dev)
                noise = self.fixed_noise
            gen = self.sampling_fn(gen, output_shape, noise=noise, model_kwargs=model_inputs, progress=True, device=dev)
            if self.undo_n1_to_1:
                gen = (gen + 1) / 2
            return {self.output: gen}
