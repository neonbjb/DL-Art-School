import os

import torch
import os.path as osp
import torchvision
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

import trainer.eval.evaluator as evaluator
from pytorch_fid import fid_score

from data.image_pair_with_corresponding_points_dataset import ImagePairWithCorrespondingPointsDataset
from utils.util import opt_get

# Uses two datasets: a "similar" and "dissimilar" dataset, each of which contains pairs of images and similar/dissimilar
# points in those datasets. Uses the provided network to compute a latent vector for both similar and dissimilar.
# Reports a score for the l2 distance of both. A properly trained network will show similar points getting closer while
# dissimilar points remain constant or get further apart.
class SinglePointPairContrastiveEval(evaluator.Evaluator):
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env)
        self.batch_sz = opt_eval['batch_size']
        self.eval_qty = opt_eval['quantity']
        assert self.eval_qty % self.batch_sz == 0
        self.similar_set = DataLoader(ImagePairWithCorrespondingPointsDataset(**opt_eval['similar_set_args']), shuffle=False, batch_size=self.batch_sz)
        self.dissimilar_set = DataLoader(ImagePairWithCorrespondingPointsDataset(**opt_eval['dissimilar_set_args']), shuffle=False, batch_size=self.batch_sz)

    def get_l2_score(self, dl):
        distances = []
        l2 = MSELoss()
        for i, data in tqdm(enumerate(dl)):
            latent1 = self.model(data['img1'], data['coords1'])
            latent2 = self.model(data['img2'], data['coords2'])
            distances.append(l2(latent1, latent2))
            if i * self.batch_sz >= self.eval_qty:
                break

        return torch.stack(distances).mean()

    def perform_eval(self):
        self.model.eval()
        print("Computing contrastive eval on similar set")
        similars = self.get_l2_score(self.similar_set)
        print("Computing contrastive eval on dissimilar set")
        dissimilars = self.get_l2_score(self.dissimilar_set)
        print(f"Eval done. val_similar_lq: {similars.item()}; val_dissimilar_l2: {dissimilars.item()}")
        return {"val_similar_l2": similars.item(), "val_dissimilar_l2": dissimilars.item()}