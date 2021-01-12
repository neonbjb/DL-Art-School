import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

from models.pixel_level_contrastive_learning.resnet_unet import UResNet50
from trainer.networks import register_model
from utils.kmeans import kmeans_predict
from utils.util import opt_get


class UResnetMaskProducer(nn.Module):
    def __init__(self, pretrained_uresnet_path, kmeans_centroid_path, mask_scales=[.125,.25,.5,1], tail_dim=512):
        super().__init__()
        _, centroids = torch.load(kmeans_centroid_path)
        self.centroids = nn.Parameter(centroids)
        self.ures = UResNet50(Bottleneck, [3,4,6,3], out_dim=tail_dim).to('cuda')
        self.mask_scales = mask_scales

        sd = torch.load(pretrained_uresnet_path)
        # An assumption is made that the state_dict came from a byol model. Strip out unnecessary weights..
        resnet_sd = {}
        for k, v in sd.items():
            if 'target_encoder.net.' in k:
                resnet_sd[k.replace('target_encoder.net.', '')] = v

        self.ures.load_state_dict(resnet_sd, strict=True)
        self.ures.eval()

    def forward(self, x):
        with torch.no_grad():
            latents = self.ures(x)
            b,c,h,w = latents.shape
            latents = latents.permute(0,2,3,1).reshape(b*h*w,c)
            masks = kmeans_predict(latents, self.centroids).float()
            masks = masks.reshape(b,1,h,w)
            interpolated_masks = {}
            for sf in self.mask_scales:
                dim_h, dim_w = int(sf*x.shape[-2]), int(sf*x.shape[-1])
                imask = F.interpolate(masks, size=(dim_h,dim_w), mode="nearest")
                interpolated_masks[dim_w] = imask.long()
            return interpolated_masks


@register_model
def register_uresnet_mask_producer(opt_net, opt):
    kw = opt_get(opt_net, ['kwargs'], {})
    return UResnetMaskProducer(**kw)
