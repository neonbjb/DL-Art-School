import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize
from tqdm import tqdm

from data.images.image_folder_dataset import ImageFolderDataset
from models.segformer.segformer import Segformer

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


def get_image_folder_dataloader(batch_size, num_workers, target_size=224, shuffle=True):
    dataset_opt = dict_to_nonedict({
        'name': 'amalgam',
        #'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\pn_coven\\cropped2'],
        #'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\imageset_1024_square_with_new'],
        #'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\imageset_256_tiled_filtered_flattened'],
        #'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\1024_test'],
        'paths': ['E:\\4k6k\\datasets\\ns_images\\imagesets\\imageset_256_full'],
        'weights': [1],
        'target_size': target_size,
        'force_multiple': 32,
        'normalize': 'imagenet',
        'scale': 1
    })
    dataset = ImageFolderDataset(dataset_opt)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


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


def get_latent_for_img(model, img, pos):
    img_t = ToTensor()(Image.open(img)).to('cuda')[:3]
    img_t = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)(img_t).unsqueeze(0)
    _, _, h, w = img_t.shape
    # Center crop img_t and resize to 224.
    d = min(h, w)
    dh, dw = (h-d)//2, (w-d)//2
    if dw != 0:
        img_t = img_t[:, :, :, dw:-dw]
        pos[1] = pos[1]-dw
    elif dh != 0:
        img_t = img_t[:, :, dh:-dh, :]
        pos[0] = pos[0]-dh
    scale = 224 / img_t.shape[-1]
    pos = (pos * scale).long()
    assert(pos.min() >= 0 and pos.max() < 224)
    img_t = img_t[:,:3,:,:]
    img_t = torch.nn.functional.interpolate(img_t, size=(224, 224), mode="area")
    latent = model(img=img_t,pos=pos)
    return latent


def produce_latent_dict(model):
    batch_size = 32
    num_workers = 4
    dataloader = get_image_folder_dataloader(batch_size, num_workers)
    id = 0
    paths = []
    latents = []
    points = []
    for batch in tqdm(dataloader):
        hq = batch['hq'].to('cuda')
        # Pull several points from every image.
        for k in range(10):
            _, _, h, _ = hq.shape
            point = torch.randint(h//4, 3*h//4, (2,)).long().to(hq.device)
            model(img=hq, pos=point)
            l = layer_hooked_value.cpu().split(1, dim=0)
            latents.extend(l)
            points.extend([point for p in range(batch_size)])
            paths.extend(batch['HQ_path'])
            id += batch_size
            if id > 10000:
                print("Saving checkpoint..")
                torch.save((latents, points, paths), '../results_segformer.pth')
                id = 0


def find_similar_latents(model, compare_fn=structural_euc_dist):
    global layer_hooked_value

    img = 'F:\\dlas\\results\\bobz.png'
    #img = 'F:\\4k6k\\datasets\\ns_images\\adrianna\\analyze\\analyze_xx\\nicky_xx.jpg'
    point=torch.tensor([154,330], dtype=torch.long, device='cuda')

    output_path = '../../../results/byol_resnet_similars'
    os.makedirs(output_path, exist_ok=True)
    imglatent = get_latent_for_img(model, img, point).squeeze().unsqueeze(0)
    _, c = imglatent.shape

    batch_size = 512
    num_workers = 1
    dataloader = get_image_folder_dataloader(batch_size, num_workers)
    id = 0
    output_batch = 1
    results = []
    result_paths = []
    results_points = []
    for batch in tqdm(dataloader):
        hq = batch['hq'].to('cuda')
        _,_,h,w = hq.shape
        point = torch.randint(h//4, 3*h//4, (2,)).long().to(hq.device)
        latent = model(img=hq, pos=point)
        compared = compare_fn(imglatent.repeat(latent.shape[0], 1), latent)
        results.append(compared.cpu())
        result_paths.extend(batch['HQ_path'])
        results_points.append(point.unsqueeze(0).repeat(batch_size,1))
        id += batch_size
        if id > 10000:
            k = 10
            results = torch.cat(results, dim=0)
            results_points = torch.cat(results_points, dim=0)
            vals, inds = torch.topk(results, k, largest=False)
            for i in inds:
                point = results_points[i]
                mag = int(results[i].item() * 100000000)
                hqr = ToTensor()(Image.open(result_paths[i])).to('cuda')
                hqr *= .5
                hqr[:,point[0]-3:point[0]+3,point[1]-3:point[1]+3] *= 2
                torchvision.utils.save_image(hqr, os.path.join(output_path, f'{mag:08}_{output_batch}_{i}.jpg'))
            results = []
            result_paths = []
            results_points = []
            id = 0


def build_kmeans():
    latents, _, _ = torch.load('../results_segformer.pth')
    latents = torch.cat(latents, dim=0).squeeze().to('cuda')[50000:] * 10000
    cluster_ids_x, cluster_centers = kmeans(latents, num_clusters=16, distance="euclidean", device=torch.device('cuda:0'))
    torch.save((cluster_ids_x, cluster_centers), '../k_means_segformer.pth')


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def use_kmeans():
    output = "../results/k_means_segformer/"
    _, centers = torch.load('../k_means_segformer.pth')
    centers = centers.to('cuda')
    batch_size = 32
    num_workers = 1
    dataloader = get_image_folder_dataloader(batch_size, num_workers, target_size=224, shuffle=True)
    denorm = UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    for i, batch in enumerate(tqdm(dataloader)):
        hq = batch['hq'].to('cuda')
        _,_,h,w = hq.shape
        point = torch.randint(h//4, 3*h//4, (2,)).long().to(hq.device)
        model(hq, point)
        l = layer_hooked_value.clone().squeeze()
        pred = kmeans_predict(l, centers)
        hq = denorm(hq * .5)
        hq[:,:,point[0]-5:point[0]+5,point[1]-5:point[1]+5] *= 2
        for b in range(pred.shape[0]):
            outpath = os.path.join(output, str(pred[b].item()))
            os.makedirs(outpath, exist_ok=True)
            torchvision.utils.save_image(hq[b], os.path.join(outpath, f'{i*batch_size+b}.png'))


if __name__ == '__main__':
    pretrained_path = '../../../experiments/segformer_contrastive.pth'
    model = Segformer().to('cuda')
    sd = torch.load(pretrained_path)
    resnet_sd = {}
    for k, v in sd.items():
        if 'target_encoder.net.' in k:
            resnet_sd[k.replace('target_encoder.net.', '')] = v
    model.load_state_dict(resnet_sd, strict=True)
    model.eval()
    register_hook(model, 'tail')

    with torch.no_grad():
        find_similar_latents(model, structural_euc_dist)
        #produce_latent_dict(model)
        #build_kmeans()
        #use_kmeans()
