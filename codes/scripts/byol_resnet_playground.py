import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize
from tqdm import tqdm
import numpy as np

import utils
from data.image_folder_dataset import ImageFolderDataset
from models.resnet_with_checkpointing import resnet50
from models.spinenet_arch import SpineNet


# Computes the structural euclidean distance between [x,y]. "Structural" here means the [h,w] dimensions are preserved
# and the distance is computed across the channel dimension.
from utils import util
from utils.options import dict_to_nonedict


def structural_euc_dist(x, y):
    diff = torch.square(x - y)
    sum = torch.sum(diff, dim=-1)
    return torch.mean(torch.sqrt(sum))


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
        'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\imageset_1024_square_with_new'],
        #'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\1024_test'],
        'weights': [1],
        'target_size': 224,
        'force_multiple': 32,
        'scale': 1
    })
    dataset = ImageFolderDataset(dataset_opt)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def _find_layer(net, layer_name):
    if type(layer_name) == str:
        modules = dict([*net.named_modules()])
        return modules.get(layer_name, None)
    elif type(layer_name) == int:
        children = [*net.children()]
        return children[layer_name]
    return None


layer_hooked_value = None
def _hook(_, __, output):
    global layer_hooked_value
    layer_hooked_value = output


def register_hook(net, layer_name):
    layer = _find_layer(net, layer_name)
    assert layer is not None, f'hidden layer ({self.layer}) not found'
    layer.register_forward_hook(_hook)


def create_latent_database(model, model_index=0):
    batch_size = 32
    num_workers = 1
    output_path = '../../results/byol_resnet_latents/'

    os.makedirs(output_path, exist_ok=True)
    dataloader = get_image_folder_dataloader(batch_size, num_workers)
    id = 0
    dict_count = 1
    latent_dict = {}
    all_paths = []
    for batch in tqdm(dataloader):
        hq = batch['hq'].to('cuda')
        latent = model(hq)[model_index]   # BYOL trainer only trains the '4' output, which is indexed at [1]. Confusing.
        for b in range(latent.shape[0]):
            im_path = batch['HQ_path'][b]
            all_paths.append(im_path)
            latent_dict[id] = latent[b].detach().cpu()
            if (id+1) % 1000 == 0:
                print("Saving checkpoint..")
                torch.save(latent_dict, os.path.join(output_path, "latent_dict_%i.pth" % (dict_count,)))
                latent_dict = {}
                torch.save(all_paths, os.path.join(output_path, "all_paths.pth"))
                dict_count += 1
            id += 1



def get_latent_for_img(model, img):
    img_t = ToTensor()(Image.open(img)).to('cuda').unsqueeze(0)
    _, _, h, w = img_t.shape
    # Center crop img_t and resize to 224.
    d = min(h, w)
    dh, dw = (h-d)//2, (w-d)//2
    if dh == 0:
        img_t = img_t[:, :, :, dw:-dw]
    else:
        img_t = img_t[:, :, dh:-dh, :]
    img_t = torch.nn.functional.interpolate(img_t, size=(224, 224), mode="area")
    model(img_t)
    latent = layer_hooked_value
    return latent


def find_similar_latents(model, compare_fn=structural_euc_dist):
    global layer_hooked_value

    img = 'F:\\4k6k\\datasets\\ns_images\\adrianna\\analyze\\analyze_xx\\yui_xx.jpg'
    #img = 'F:\\4k6k\\datasets\\ns_images\\adrianna\\analyze\\analyze_xx\\nicky_xx.jpg'
    output_path = '../../results/byol_resnet_similars'
    os.makedirs(output_path, exist_ok=True)
    imglatent = get_latent_for_img(model, img)
    _, c, h, w = imglatent.shape

    batch_size = 32
    num_workers = 1
    dataloader = get_image_folder_dataloader(batch_size, num_workers)
    id = 0
    results = []
    for batch in tqdm(dataloader):
        hq = batch['hq'].to('cuda')
        model(hq)
        latent = layer_hooked_value
        for b in range(latent.shape[0]):
            im_path = batch['HQ_path'][b]
            results.append((im_path, compare_fn(imglatent, latent[b].unsqueeze(0)).item()))
            id += 1
        if id > 2000:
            break
    results.sort(key=lambda x: x[1])
    for i in range(50):
        mag = results[i][1]
        shutil.copy(results[i][0], os.path.join(output_path, f'{i}_{mag}.jpg'))


if __name__ == '__main__':
    pretrained_path = '../../experiments/resnet_byol_diffframe_69k.pth'
    model = resnet50(pretrained=False).to('cuda')
    sd = torch.load(pretrained_path)
    resnet_sd = {}
    for k, v in sd.items():
        if 'target_encoder.net.' in k:
            resnet_sd[k.replace('target_encoder.net.', '')] = v
    model.load_state_dict(resnet_sd, strict=True)
    model.eval()
    register_hook(model, 'avgpool')

    with torch.no_grad():
        find_similar_latents(model, structural_euc_dist)
