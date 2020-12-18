import os

import torch
import os.path as osp
import torchvision
import trainer.eval.evaluator as evaluator
from pytorch_fid import fid_score


# Evaluate that generates uniform noise to feed into a generator, then calculates a FID score on the results.
class StyleTransferEvaluator(evaluator.Evaluator):
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env)
        self.batches_per_eval = opt_eval['batches_per_eval']
        self.batch_sz = opt_eval['batch_size']
        self.im_sz = opt_eval['image_size']
        self.fid_real_samples = opt_eval['real_fid_path']
        self.gen_output_index = opt_eval['gen_index'] if 'gen_index' in opt_eval.keys() else 0

    def perform_eval(self):
        fid_fake_path = osp.join(self.env['base_path'], "../../models", "fid", str(self.env["step"]))
        os.makedirs(fid_fake_path, exist_ok=True)
        counter = 0
        for i in range(self.batches_per_eval):
            batch = torch.FloatTensor(self.batch_sz, 3, self.im_sz, self.im_sz).uniform_(0., 1.).to(self.env['device'])
            gen = self.model(batch)
            if not isinstance(gen, list) and not isinstance(gen, tuple):
                gen = [gen]
            gen = gen[self.gen_output_index]
            for b in range(self.batch_sz):
                torchvision.utils.save_image(gen[b], osp.join(fid_fake_path, "%i_.png" % (counter)))
                counter += 1

        return {"fid": fid_score.calculate_fid_given_paths([self.fid_real_samples, fid_fake_path], self.batch_sz, True,
                                                           2048)}
