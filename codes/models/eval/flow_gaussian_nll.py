import os

import torch
import os.path as osp
import torchvision
from torch.utils.data import DataLoader

import models.eval.evaluator as evaluator
from pytorch_fid import fid_score


# Evaluate how close to true Gaussian a flow network predicts in a "normal" pass given a LQ/HQ image pair.
from data.image_folder_dataset import ImageFolderDataset
from models.archs.srflow_orig.flow import GaussianDiag


class FlowGaussianNll(evaluator.Evaluator):
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env)
        self.batch_sz = opt_eval['batch_size']
        self.dataset = ImageFolderDataset(opt_eval['dataset'])
        self.dataloader = DataLoader(self.dataset, self.batch_sz)

    def perform_eval(self):
        total_zs = 0
        z_loss = 0
        with torch.no_grad():
            for batch in self.dataloader:
                z, _, _ = self.model(gt=batch['GT'],
                                     lr=batch['LQ'],
                                     epses=[],
                                     reverse=False,
                                     add_gt_noise=False)
                for z_ in z:
                    z_loss += GaussianDiag.logp(None, None, z_).mean()
                    total_zs += 1
        return {"gaussian_diff": z_loss / total_zs}
