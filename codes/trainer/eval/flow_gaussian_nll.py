import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import trainer.eval.evaluator as evaluator

# Evaluate how close to true Gaussian a flow network predicts in a "normal" pass given a LQ/HQ image pair.
from data.images.image_folder_dataset import ImageFolderDataset
from models.image_generation.srflow.flow import GaussianDiag


class FlowGaussianNll(evaluator.Evaluator):
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env, uses_all_ddp=False)
        self.batch_sz = opt_eval['batch_size']
        self.dataset = ImageFolderDataset(opt_eval['dataset'])
        self.dataloader = DataLoader(self.dataset, self.batch_sz)

    def perform_eval(self):
        total_zs = 0
        z_loss = 0
        self.model.eval()
        with torch.no_grad():
            print("Evaluating FlowGaussianNll..")
            for batch in tqdm(self.dataloader):
                dev = self.env['device']
                z, _, _ = self.model(gt=batch['hq'].to(dev),
                                     lr=batch['lq'].to(dev),
                                     epses=[],
                                     reverse=False,
                                     add_gt_noise=False)
                for z_ in z:
                    z_loss += GaussianDiag.logp(None, None, z_).mean()
                    total_zs += 1
        self.model.train()
        return {"gaussian_diff": z_loss / total_zs}
