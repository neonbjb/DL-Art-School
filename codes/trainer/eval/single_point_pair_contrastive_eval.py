import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

import trainer.eval.evaluator as evaluator

from data.images.image_pair_with_corresponding_points_dataset import ImagePairWithCorrespondingPointsDataset


# Uses two datasets: a "similar" and "dissimilar" dataset, each of which contains pairs of images and similar/dissimilar
# points in those datasets. Uses the provided network to compute a latent vector for both similar and dissimilar.
# Reports a score for the l2 distance of both. A properly trained network will show similar points getting closer while
# dissimilar points remain constant or get further apart.
class SinglePointPairContrastiveEval(evaluator.Evaluator):
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env, uses_all_ddp=False)
        self.batch_sz = opt_eval['batch_size']
        self.eval_qty = opt_eval['quantity']
        assert self.eval_qty % self.batch_sz == 0
        self.similar_set = DataLoader(ImagePairWithCorrespondingPointsDataset(opt_eval['similar_set_args']), shuffle=False, batch_size=self.batch_sz)
        self.dissimilar_set = DataLoader(ImagePairWithCorrespondingPointsDataset(opt_eval['dissimilar_set_args']), shuffle=False, batch_size=self.batch_sz)
        # Hack to make this work with the BYOL generator. TODO: fix
        self.model = self.model.online_encoder.net

    def get_l2_score(self, dl, dev):
        distances = []
        l2 = MSELoss()
        for i, data in tqdm(enumerate(dl)):
            latent1 = self.model(img=data['img1'].to(dev), pos=torch.stack(data['coords1'], dim=1).to(dev))
            latent2 = self.model(img=data['img2'].to(dev), pos=torch.stack(data['coords2'], dim=1).to(dev))
            distances.append(l2(latent1, latent2))
            if i * self.batch_sz >= self.eval_qty:
                break

        return torch.stack(distances).mean()

    def perform_eval(self):
        self.model.eval()
        with torch.no_grad():
            dev = next(self.model.parameters()).device
            print("Computing contrastive eval on similar set")
            similars = self.get_l2_score(self.similar_set, dev)
            print("Computing contrastive eval on dissimilar set")
            dissimilars = self.get_l2_score(self.dissimilar_set, dev)
            diff = dissimilars.item() - similars.item()
            print(f"Eval done. val_similar_lq: {similars.item()}; val_dissimilar_l2: {dissimilars.item()}; val_diff: {diff}")
        self.model.train()
        return {"val_similar_l2": similars.item(), "val_dissimilar_l2": dissimilars.item(), "val_diff": diff}


if __name__ == '__main__':
    model = Segformer(1024, 4).cuda()
    eval = SinglePointPairContrastiveEval(model, {
        'batch_size': 8,
        'quantity': 32,
        'similar_set_args': {
            'path': 'E:\\4k6k\\datasets\\ns_images\\segformer_validation\\similar',
            'size': 256
        },
        'dissimilar_set_args': {
            'path': 'E:\\4k6k\\datasets\\ns_images\\segformer_validation\\dissimilar',
            'size': 256
        },
    }, {})
    eval.perform_eval()
