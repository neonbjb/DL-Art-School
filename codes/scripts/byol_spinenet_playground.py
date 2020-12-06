import os
import shutil

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.image_folder_dataset import ImageFolderDataset
from models.archs.spinenet_arch import SpineNet


# Computes the structural euclidean distance between [x,y]. "Structural" here means the [h,w] dimensions are preserved
# and the distance is computed across the channel dimension.
def structural_euc_dist(x, y):
    diff = torch.square(x - y)
    sum = torch.sum(diff, dim=1)
    return torch.sqrt(sum)


def cosine_similarity(x, y):
    return nn.CosineSimilarity()(x, y)   # probably better to just use this class to perform the calc. Just left this here to remind myself.


def im_norm(x):
    return (((x - torch.mean(x, dim=(2,3)).reshape(-1,1,1,1)) / torch.std(x, dim=(2,3)).reshape(-1,1,1,1)) * .5) + .5


def get_image_folder_dataloader(batch_size, num_workers):
    dataset_opt = {
        'name': 'amalgam',
        'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\imageset_1024_square_with_new'],
        'weights': [1],
        'target_size': 512,
        'force_multiple': 32,
        'scale': 1
    }
    dataset = ImageFolderDataset(dataset_opt)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def create_latent_database(model):
    batch_size = 8
    num_workers = 1
    output_path = '../../results/byol_spinenet_latents/'

    os.makedirs(output_path, exist_ok=True)
    dataloader = get_image_folder_dataloader(batch_size, num_workers)
    id = 0
    latent_dict = {}
    for batch in tqdm(dataloader):
        hq = batch['hq'].to('cuda:1')
        latent = model(hq)[1]   # BYOL trainer only trains the '4' output, which is indexed at [1]. Confusing.
        for b in range(latent.shape[0]):
            shutil.copy(batch[b]['HQ_path'], os.path.join(output_path, "%i.jpg" % (id,)))
            latent_dict[id] = latent[b].detach().cpu()
            if id % 100 == 0:
                print("Saving checkpoint..")
                torch.save(latent_dict, "latent_dict.pth")
            id += 1


def explore_latent_results(model):
    batch_size = 8
    num_workers = 1
    output_path = '../../results/byol_spinenet_explore_latents/'

    os.makedirs(output_path, exist_ok=True)
    dataloader = get_image_folder_dataloader(batch_size, num_workers)
    id = 0
    for batch in tqdm(dataloader):
        hq = batch['hq'].to('cuda:1')
        latent = model(hq)[1]   # BYOL trainer only trains the '4' output, which is indexed at [1]. Confusing.
        # This operation works by computing the distance of every structural index from the center and using that
        # as a "heatmap".
        b, c, h, w = latent.shape
        center = latent[:, :, h//2, w//2].unsqueeze(-1).unsqueeze(-1)
        centers = center.repeat(1, 1, h, w)
        dist = structural_euc_dist(latent, centers).unsqueeze(1)
        dist = im_norm(dist)
        torchvision.utils.save_image(dist, os.path.join(output_path, "%i.png" % id))
        id += 1


if __name__ == '__main__':
    pretrained_path = '../../experiments/spinenet49_imgset_byol.pth'

    model = SpineNet('49', in_channels=3, use_input_norm=True).to('cuda:1')
    model.load_state_dict(torch.load(pretrained_path), strict=True)
    model.eval()

    with torch.no_grad():
        explore_latent_results(model)