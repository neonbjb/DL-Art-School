import torch
import torchvision
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import trainer.eval.evaluator as evaluator
from data import create_dataset
from models.vqvae.kmeans_mask_producer import UResnetMaskProducer
from utils.util import opt_get


class CategorizationLossEvaluator(evaluator.Evaluator):
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env)
        self.batch_sz = opt_eval['batch_size']
        assert self.batch_sz is not None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.dataset = create_dataset(opt_eval['dataset'])
        self.dataloader = DataLoader(self.dataset, self.batch_sz, shuffle=False, num_workers=4)
        self.gen_output_index = opt_eval['gen_index'] if 'gen_index' in opt_eval.keys() else 0
        self.masking = opt_get(opt_eval, ['masking'], False)
        if self.masking:
            self.mask_producer = UResnetMaskProducer(pretrained_uresnet_path= '../experiments/train_imagenet_pixpro_resnet/models/66500_generator.pth',
                                                     kmeans_centroid_path='../experiments/k_means_uresnet_imagenet_256.pth',
                                                     mask_scales=[.03125, .0625, .125, .25, .5, 1.0],
                                                     tail_dim=256).to('cuda')

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target[None])

            res = []
            for k in topk:
                correct_k = correct[:k].flatten().sum(dtype=torch.float32)
                res.append(correct_k * (100.0 / batch_size))
            return res

    def perform_eval(self):
        counter = 0.0
        ce_loss = 0.0
        top_5_acc = 0.0
        top_1_acc = 0.0

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                hq, labels = batch['hq'], batch['labels']
                hq = hq.to(self.env['device'])
                labels = labels.to(self.env['device'])
                if self.masking:
                    masks = self.mask_producer(hq)
                    logits = self.model(hq, masks)
                else:
                    logits = self.model(hq)
                if not isinstance(logits, list) and not isinstance(logits, tuple):
                    logits = [logits]
                logits = logits[self.gen_output_index]
                ce_loss += torch.nn.functional.cross_entropy(logits, labels).detach()
                t1, t5 = self.accuracy(logits, labels, (1, 5))
                top_1_acc += t1.detach()
                top_5_acc += t5.detach()
                counter += len(hq) / self.batch_sz
        self.model.train()

        return {"val_cross_entropy": ce_loss / counter,
                "top_5_accuracy": top_5_acc / counter,
                "top_1_accuracy": top_1_acc / counter }


if __name__ == '__main__':
    from torchvision.models import resnet50
    model = resnet50(pretrained=True).to('cuda')
    opt = {
        'batch_size': 128,
        'gen_index': 0,
        'masking': False
    }
    env = {
        'device': 'cuda',

    }
    eval = CategorizationLossEvaluator(model, opt, env)
    print(eval.perform_eval())
