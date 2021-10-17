import torch

from models.diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule
from models.diffusion.resample import create_named_schedule_sampler, LossAwareSampler
from models.diffusion.respace import space_timesteps, SpacedDiffusion
from trainer.inject import Injector
from utils.util import opt_get


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

    def forward(self, state):
        gen = self.env['generators'][self.opt['generator']]
        hq = state[self.input]
        model_inputs = {k: state[v] for k, v in self.model_input_keys.items()}
        t, weights = self.schedule_sampler.sample(hq.shape[0], hq.device)
        diffusion_outputs = self.diffusion.training_losses(gen, hq, t, model_kwargs=model_inputs)
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, diffusion_outputs['losses'])

        if len(self.extra_model_output_keys) > 0:
            assert(len(self.extra_model_output_keys) == len(diffusion_outputs['extra_outputs']))
            out = {k: v for k, v in zip(self.extra_model_output_keys, diffusion_outputs['extra_outputs'])}
        else:
            out = {}
        out.update({self.output: diffusion_outputs['mse'],
                self.output_variational_bounds_key: diffusion_outputs['vb'],
                self.output_x_start_key: diffusion_outputs['x_start_predicted']})
        return out


class AutoregressiveGaussianDiffusionInjector(Injector):
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
        self.model_output_keys = opt['model_output_keys']
        self.model_eps_pred_key = opt['prediction_key']

    def forward(self, state):
        gen = self.env['generators'][self.opt['generator']]
        hq = state[self.input]
        model_inputs = {k: state[v] for k, v in self.model_input_keys.items()}
        t, weights = self.schedule_sampler.sample(hq.shape[0], hq.device)
        diffusion_outputs = self.diffusion.autoregressive_training_losses(gen, hq, t, self.model_output_keys,
                                                                          self.model_eps_pred_key,
                                                                          model_kwargs=model_inputs)
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, diffusion_outputs['losses'])
        outputs = {k: diffusion_outputs[k] for k in self.model_output_keys}
        outputs.update({self.output: diffusion_outputs['mse'],
                self.output_variational_bounds_key: diffusion_outputs['vb'],
                self.output_x_start_key: diffusion_outputs['x_start_predicted']})
        return outputs


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
                output_shape = (self.output_batch_size, 1, model_inputs['spectrogram'].shape[-1]*256)
                dev = model_inputs['spectrogram'].device
            elif 'discrete_spectrogram' in model_inputs.keys():
                output_shape = (self.output_batch_size, 1, model_inputs['discrete_spectrogram'].shape[-1]*1024)
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
