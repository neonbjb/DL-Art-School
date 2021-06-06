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

    def forward(self, state):
        gen = self.env['generators'][self.opt['generator']]
        hq = state[self.input]
        model_inputs = {k: state[v] for k, v in self.model_input_keys.items()}
        t, weights = self.schedule_sampler.sample(hq.shape[0], hq.device)
        diffusion_outputs = self.diffusion.training_losses(gen, hq, t, model_kwargs=model_inputs)
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, diffusion_outputs['losses'])
        return {self.output: diffusion_outputs['mse'],
                self.output_variational_bounds_key: diffusion_outputs['vb'],
                self.output_x_start_key: diffusion_outputs['x_start_predicted']}


# Performs inference using a network trained to predict a reverse diffusion process, which nets a image.
class GaussianDiffusionInferenceInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.generator = opt['generator']
        self.output_shape = opt['output_shape']
        opt['diffusion_args']['betas'] = get_named_beta_schedule(**opt['beta_schedule'])
        opt['diffusion_args']['use_timesteps'] = space_timesteps(opt['beta_schedule']['num_diffusion_timesteps'],
                                                                 [opt_get(opt, ['respaced_timestep_spacing'], opt['beta_schedule']['num_diffusion_timesteps'])])
        self.diffusion = SpacedDiffusion(**opt['diffusion_args'])
        self.model_input_keys = opt_get(opt, ['model_input_keys'], [])

    def forward(self, state):
        gen = self.env['generators'][self.opt['generator']]
        batch_size = self.output_shape[0]
        model_inputs = {k: state[v][:batch_size] for k, v in self.model_input_keys.items()}
        gen.eval()
        with torch.no_grad():
            gen = self.diffusion.p_sample_loop(gen, self.output_shape, model_kwargs=model_inputs)
            return {self.output: gen}
