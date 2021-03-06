import os
from random import shuffle

import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.resnet import Bottleneck
from tqdm import tqdm

from data.image_folder_dataset import ImageFolderDataset
from models.pixel_level_contrastive_learning.resnet_unet_3 import UResNet50_3

# Computes the structural euclidean distance between [x,y]. "Structural" here means the [h,w] dimensions are preserved
# and the distance is computed across the channel dimension.
from utils.kmeans import kmeans, kmeans_predict
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


def get_image_folder_dataloader(batch_size, num_workers, target_size=256):
    dataset_opt = dict_to_nonedict({
        'name': 'amalgam',
        #'paths': ['F:\\4k6k\\datasets\\images\\imagenet_2017\\train'],
        #'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\imageset_1024_square_with_new'],
        'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\imageset_256_full'],
        #'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\1024_test'],
        'weights': [1],
        'target_size': target_size,
        'force_multiple': 32,
        'scale': 1
    })
    dataset = ImageFolderDataset(dataset_opt)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def produce_latent_dict(model, basename):
    batch_size = 64
    num_workers = 4
    dataloader = get_image_folder_dataloader(batch_size, num_workers)
    id = 0
    paths = []
    latents = []
    prob = None
    for batch in tqdm(dataloader):
        hq = batch['hq'].to('cuda')
        l = model(hq)
        b, c, h, w = l.shape
        dim = b*h*w
        l = l.permute(0,2,3,1).reshape(dim, c).cpu()
        # extract a random set of 10 latents from each image
        if prob is None:
            prob = torch.full((dim,), 1/(dim))
        l = l[prob.multinomial(num_samples=100, replacement=False)].split(1, dim=0)
        latents.extend(l)
        paths.extend(batch['HQ_path'])
        id += batch_size
        if id > 5000:
            print("Saving checkpoint..")
            torch.save((latents, paths), f'../{basename}_latent_dict.pth')
            id = 0


def build_kmeans(basename):
    latents, _ = torch.load(f'../{basename}_latent_dict.pth')
    shuffle(latents)
    latents = torch.cat(latents, dim=0).to('cuda')
    cluster_ids_x, cluster_centers = kmeans(latents, num_clusters=8, distance="euclidean", device=torch.device('cuda:0'), tol=0, iter_limit=5000, gravity_limit_per_iter=1000)
    torch.save((cluster_ids_x, cluster_centers), f'../{basename}_k_means_centroids.pth')


def use_kmeans(basename):
    output_path = f'../results/{basename}_kmeans_viz'
    _, centers = torch.load(f'../{basename}_k_means_centroids.pth')
    centers = centers.to('cuda')
    batch_size = 8
    num_workers = 0
    dataloader = get_image_folder_dataloader(batch_size, num_workers, target_size=256)
    colormap = cm.get_cmap('viridis', 8)
    os.makedirs(output_path, exist_ok=True)
    for i, batch in enumerate(tqdm(dataloader)):
        hq = batch['hq'].to('cuda')
        l = model(hq)
        b, c, h, w = l.shape
        dim = b*h*w
        l = l.permute(0,2,3,1).reshape(dim,c)
        pred = kmeans_predict(l, centers)
        pred = pred.reshape(b,h,w)
        img = torch.tensor(colormap(pred[:, :, :].detach().cpu().numpy()))
        scale = hq.shape[-2] / h
        torchvision.utils.save_image(torch.nn.functional.interpolate(img.permute(0,3,1,2), scale_factor=scale, mode="nearest"),
                                     f"{output_path}/{i}_categories.png")
        torchvision.utils.save_image(hq, f"{output_path}/{i}_hq.png")


if __name__ == '__main__':
    pretrained_path = '../experiments/uresnet_pixpro4_imgset.pth'
    basename = 'uresnet_pixpro4'
    model = UResNet50_3(Bottleneck, [3,4,6,3], out_dim=64).to('cuda')
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
        #produce_latent_dict(model, basename)
        #uild_kmeans(basename)
        use_kmeans(basename)
