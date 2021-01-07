import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.resnet import Bottleneck
from torchvision.transforms import ToTensor, Resize
from tqdm import tqdm
import numpy as np

import utils
from data.image_folder_dataset import ImageFolderDataset
from models.pixel_level_contrastive_learning.resnet_unet import UResNet50
from models.resnet_with_checkpointing import resnet50
from models.spinenet_arch import SpineNet


# Computes the structural euclidean distance between [x,y]. "Structural" here means the [h,w] dimensions are preserved
# and the distance is computed across the channel dimension.
from scripts.byol.byol_spinenet_playground import find_similar_latents, create_latent_database
from utils import util
from utils.options import dict_to_nonedict


def structural_euc_dist(x, y):
    diff = torch.square(x - y)
    sum = torch.sum(diff, dim=-1)
    return torch.sqrt(sum)


def cosine_similarity(x, y):
    x = norm(x)
    y = norm(y)
    return -nn.CosineSimilarity()(x, y)   # probably better to just use this class to perform the calc. Just left this here to remind myself.


def key_value_difference(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def norm(x):
    sh = x.shape
    sh_r = tuple([sh[i] if i != len(sh)-1 else 1 for i in range(len(sh))])
    return (x - torch.mean(x, dim=-1).reshape(sh_r)) / torch.std(x, dim=-1).reshape(sh_r)


def im_norm(x):
    return (((x - torch.mean(x, dim=(2,3)).reshape(-1,1,1,1)) / torch.std(x, dim=(2,3)).reshape(-1,1,1,1)) * .5) + .5


def get_image_folder_dataloader(batch_size, num_workers):
    dataset_opt = dict_to_nonedict({
        'name': 'amalgam',
        'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\imageset_256_full'],
        #'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\1024_test'],
        'weights': [1],
        'target_size': 256,
        'force_multiple': 32,
        'scale': 1
    })
    dataset = ImageFolderDataset(dataset_opt)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def produce_latent_dict(model):
    batch_size = 32
    num_workers = 4
    dataloader = get_image_folder_dataloader(batch_size, num_workers)
    id = 0
    paths = []
    latents = []
    for batch in tqdm(dataloader):
        hq = batch['hq'].to('cuda')
        l = model(hq).cpu().split(1, dim=0)
        latents.extend(l)
        paths.extend(batch['HQ_path'])
        id += batch_size
        if id > 1000:
            print("Saving checkpoint..")
            torch.save((latents, paths), '../results.pth')
            id = 0


if __name__ == '__main__':
    pretrained_path = '../experiments/uresnet_pixpro_attempt2.pth'
    model = UResNet50(Bottleneck, [3,4,6,3], out_dim=512).to('cuda')
    sd = torch.load(pretrained_path)
    resnet_sd = {}
    for k, v in sd.items():
        if 'target_encoder.net.' in k:
            resnet_sd[k.replace('target_encoder.net.', '')] = v
    model.load_state_dict(resnet_sd, strict=True)
    model.eval()

    with torch.no_grad():
        #find_similar_latents(model, 0, 8, structural_euc_dist)
        #create_latent_database(model, batch_size=32)
        produce_latent_dict(model)
