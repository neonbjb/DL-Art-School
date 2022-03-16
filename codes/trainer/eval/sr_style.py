import os

import torch
import os.path as osp
import torchvision
from torch.utils.data import BatchSampler

import trainer.eval.evaluator as evaluator
from pytorch_fid import fid_score


# Evaluate that feeds a LR structure into the input, then calculates a FID score on the results added to
# the interpolated LR structure.
from data.images.stylegan2_dataset import Stylegan2Dataset


class SrStyleTransferEvaluator(evaluator.Evaluator):
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env, uses_all_ddp=False)
        self.batches_per_eval = opt_eval['batches_per_eval']
        self.batch_sz = opt_eval['batch_size']
        self.im_sz = opt_eval['image_size']
        self.scale = opt_eval['scale']
        self.fid_real_samples = opt_eval['real_fid_path']
        self.embedding_generator = opt_eval['embedding_generator']
        self.gen_output_index = opt_eval['gen_index'] if 'gen_index' in opt_eval.keys() else 0
        self.dataset = Stylegan2Dataset({'path': self.fid_real_samples,
                                         'target_size': self.im_sz,
                                         'aug_prob': 0,
                                         'transparent': False})
        self.sampler = BatchSampler(self.dataset, self.batch_sz, False)

    def perform_eval(self):
        embedding_generator = self.env['generators'][self.embedding_generator]
        fid_fake_path = osp.join(self.env['base_path'], "../../models", "fid_fake", str(self.env["step"]))
        os.makedirs(fid_fake_path, exist_ok=True)
        fid_real_path = osp.join(self.env['base_path'], "../../models", "fid_real", str(self.env["step"]))
        os.makedirs(fid_real_path, exist_ok=True)
        counter = 0
        for batch in self.sampler:
            noise = torch.FloatTensor(self.batch_sz, 3, self.im_sz, self.im_sz).uniform_(0., 1.).to(self.env['device'])
            batch_hq = [e['hq'] for e in batch]
            batch_hq = torch.stack(batch_hq, dim=0).to(self.env['device'])
            resized_batch = torch.nn.functional.interpolate(batch_hq, scale_factor=1/self.scale, mode="area")
            embedding = embedding_generator(resized_batch)
            gen = self.model(noise, embedding)
            if not isinstance(gen, list) and not isinstance(gen, tuple):
                gen = [gen]
            gen = gen[self.gen_output_index]
            out = gen + torch.nn.functional.interpolate(resized_batch, scale_factor=self.scale, mode='bilinear')
            for b in range(self.batch_sz):
                torchvision.utils.save_image(out[b], osp.join(fid_fake_path, "%i_.png" % (counter)))
                torchvision.utils.save_image(batch_hq[b], osp.join(fid_real_path, "%i_.png" % (counter)))
                counter += 1

        return {"fid": fid_score.calculate_fid_given_paths([fid_real_path, fid_fake_path], self.batch_sz, True,
                                                           2048)}
