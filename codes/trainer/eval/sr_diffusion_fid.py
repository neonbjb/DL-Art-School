import os
import torch
import os.path as osp
import torchvision
from torch.nn.functional import interpolate
from tqdm import tqdm

import trainer.eval.evaluator as evaluator

from pytorch_fid import fid_score
from data import create_dataset
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

from trainer.injectors.gaussian_diffusion_injector import GaussianDiffusionInferenceInjector
from utils.util import opt_get


# Performs a FID evaluation on a diffusion network
class SrDiffusionFidEvaluator(evaluator.Evaluator):
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env)
        self.batch_sz = opt_eval['batch_size']
        self.fid_batch_size = opt_get(opt_eval, ['fid_batch_size'], 64)
        assert self.batch_sz is not None
        self.dataset = create_dataset(opt_eval['dataset'])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.sampler = DistributedSampler(self.dataset, shuffle=False, drop_last=True)
        else:
            self.sampler = SequentialSampler(self.dataset)
        self.fid_real_samples = opt_eval['dataset']['paths']  # This is assumed to exist for the given dataset.
        assert isinstance(self.fid_real_samples, str)
        self.gd = GaussianDiffusionInferenceInjector(opt_eval['diffusion_params'], env)
        self.out_key = opt_eval['diffusion_params']['out']

    def perform_eval(self):
        # Attempt to make the dataset deterministic.
        self.dataset.reset_random()
        dataloader = DataLoader(self.dataset, self.batch_sz, sampler=self.sampler, num_workers=0)

        fid_fake_path = osp.join(self.env['base_path'], "..", "fid", str(self.env["step"]))
        os.makedirs(fid_fake_path, exist_ok=True)
        counter = 0
        for batch in tqdm(dataloader):
            batch = {k: v.to(self.env['device']) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            gen = self.gd(batch)[self.out_key]

            # All gather if we're in distributed mode.
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                gather_list = [torch.zeros_like(gen) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gather_list, gen)
                gen = torch.cat(gather_list, dim=0)

            if self.env['rank'] <= 0:
                for g in gen:
                    torchvision.utils.save_image(g, osp.join(fid_fake_path, f"{counter}.png"))
                    counter += 1

        if self.env['rank'] <= 0:
            return {"fid": fid_score.calculate_fid_given_paths([self.fid_real_samples, fid_fake_path], self.fid_batch_size,
                                                               True, 2048)}
        else:
            return {}
