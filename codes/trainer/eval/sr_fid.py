import os
import torch
import os.path as osp
import torchvision
from torch.nn.functional import interpolate
from tqdm import tqdm

import trainer.eval.evaluator as evaluator

from pytorch_fid import fid_score
from data import create_dataset
from torch.utils.data import DataLoader


# Computes the SR FID score for a network, which is a FID score that attempts to account for structural changes the
# generator might make from the source image.
class SrFidEvaluator(evaluator.Evaluator):
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env, uses_all_ddp=False)
        self.batch_sz = opt_eval['batch_size']
        assert self.batch_sz is not None
        self.dataset = create_dataset(opt_eval['dataset'])
        self.scale = opt_eval['scale']
        self.fid_real_samples = opt_eval['dataset']['paths']  # This is assumed to exist for the given dataset.
        assert isinstance(self.fid_real_samples, str)
        self.dataloader = DataLoader(self.dataset, self.batch_sz, shuffle=False, num_workers=1)
        self.gen_output_index = opt_eval['gen_index'] if 'gen_index' in opt_eval.keys() else 0

    def perform_eval(self):
        fid_fake_path = osp.join(self.env['base_path'], "..", "sr_fid", str(self.env["step"]))
        os.makedirs(fid_fake_path, exist_ok=True)
        counter = 0
        for batch in tqdm(self.dataloader):
            lq = batch['lq'].to(self.env['device'])
            gen = self.model(lq)
            if not isinstance(gen, list) and not isinstance(gen, tuple):
                gen = [gen]
            gen = gen[self.gen_output_index]

            # Remove low-frequency differences
            gen_lf = interpolate(interpolate(gen, scale_factor=1/self.scale, mode="area"), scale_factor=self.scale,
                                         mode="nearest")
            gen_hf = gen - gen_lf
            hq_lf = interpolate(lq, scale_factor=self.scale, mode="nearest")
            hq_gen_hf_applied = hq_lf + gen_hf

            for b in range(self.batch_sz):
                torchvision.utils.save_image(hq_gen_hf_applied[b], osp.join(fid_fake_path, "%i_.png" % (counter)))
                counter += 1

        return {"sr_fid": fid_score.calculate_fid_given_paths([self.fid_real_samples, fid_fake_path], self.batch_sz, True,
                                                           2048)}


# A "normal" FID computation from a generator that takes LR inputs. Does not account for structural differences at all.
class FidForStructuralNetsEvaluator(evaluator.Evaluator):
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env)
        self.batch_sz = opt_eval['batch_size']
        assert self.batch_sz is not None
        self.dataset = create_dataset(opt_eval['dataset'])
        self.scale = opt_eval['scale']
        self.fid_real_samples = opt_eval['dataset']['paths']  # This is assumed to exist for the given dataset.
        assert isinstance(self.fid_real_samples, str)
        self.dataloader = DataLoader(self.dataset, self.batch_sz, shuffle=False, num_workers=1)
        self.gen_output_index = opt_eval['gen_index'] if 'gen_index' in opt_eval.keys() else 0

    def perform_eval(self):
        fid_fake_path = osp.join(self.env['base_path'], "..", "fid", str(self.env["step"]))
        os.makedirs(fid_fake_path, exist_ok=True)
        counter = 0
        for batch in tqdm(self.dataloader):
            lq = batch['lq'].to(self.env['device'])
            gen = self.model(lq)
            if not isinstance(gen, list) and not isinstance(gen, tuple):
                gen = [gen]
            gen = gen[self.gen_output_index]

            for b in range(self.batch_sz):
                torchvision.utils.save_image(gen[b], osp.join(fid_fake_path, "%i_.png" % (counter)))
                counter += 1

        return {"fid": fid_score.calculate_fid_given_paths([self.fid_real_samples, fid_fake_path], self.batch_sz, True,
                                                           2048)}