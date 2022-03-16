import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data.images.image_folder_dataset import ImageFolderDataset
from models.classifiers.resnet_with_checkpointing import resnet50

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


def get_latent_for_img(model, img):
    img_t = ToTensor()(Image.open(img)).to('cuda').unsqueeze(0)
    _, _, h, w = img_t.shape
    # Center crop img_t and resize to 224.
    d = min(h, w)
    dh, dw = (h-d)//2, (w-d)//2
    if dw != 0:
        img_t = img_t[:, :, :, dw:-dw]
    elif dh != 0:
        img_t = img_t[:, :, dh:-dh, :]
    img_t = img_t[:,:3,:,:]
    img_t = torch.nn.functional.interpolate(img_t, size=(224, 224), mode="area")
    model(img_t)
    latent = layer_hooked_value
    return latent


def produce_latent_dict(model):
    batch_size = 32
    num_workers = 4
    dataloader = get_image_folder_dataloader(batch_size, num_workers)
    id = 0
    paths = []
    latents = []
    for batch in tqdm(dataloader):
        hq = batch['hq'].to('cuda')
        hq = F.interpolate(F.interpolate(hq, size=(16,16), mode='bilinear'), size=(224,244))
        model(hq)
        l = layer_hooked_value.cpu().split(1, dim=0)
        latents.extend(l)
        paths.extend(batch['HQ_path'])
        id += batch_size
        if id > 10000:
            print("Saving checkpoint..")
            torch.save((latents, paths), '../results_instance_resnet.pth')
            id = 0


def find_similar_latents(model, compare_fn=structural_euc_dist):
    global layer_hooked_value

    img = 'D:\\dlas\\results\\bobz.png'
    #img = 'F:\\4k6k\\datasets\\ns_images\\adrianna\\analyze\\analyze_xx\\nicky_xx.jpg'
    output_path = '../../../results/byol_resnet_similars'
    os.makedirs(output_path, exist_ok=True)
    imglatent = get_latent_for_img(model, img).squeeze().unsqueeze(0)
    _, c = imglatent.shape

    batch_size = 512
    num_workers = 8
    dataloader = get_image_folder_dataloader(batch_size, num_workers)
    id = 0
    output_batch = 1
    results = []
    result_paths = []
    for batch in tqdm(dataloader):
        hq = batch['hq'].to('cuda')
        model(hq)
        latent = layer_hooked_value.clone().squeeze()
        compared = compare_fn(imglatent.repeat(latent.shape[0], 1), latent)
        results.append(compared.cpu())
        result_paths.extend(batch['HQ_path'])
        id += batch_size
        if id > 10000:
            k = 200
            results = torch.cat(results, dim=0)
            vals, inds = torch.topk(results, k, largest=False)
            for i in inds:
                mag = int(results[i].item() * 1000)
                shutil.copy(result_paths[i], os.path.join(output_path, f'{mag:05}_{output_batch}_{i}.jpg'))
            results = []
            result_paths = []
            id = 0


def build_kmeans():
    latents, _ = torch.load('../results_instance_resnet.pth')
    latents = torch.cat(latents, dim=0).squeeze().to('cuda')
    cluster_ids_x, cluster_centers = kmeans(latents, num_clusters=8, distance="euclidean", device=torch.device('cuda:0'))
    torch.save((cluster_ids_x, cluster_centers), '../k_means_instance_resnet.pth')


def use_kmeans():
    output = "../results/k_means_instance_resnet/"
    _, centers = torch.load('../k_means_instance_resnet.pth')
    batch_size = 32
    num_workers = 1
    dataloader = get_image_folder_dataloader(batch_size, num_workers, target_size=224, shuffle=False)
    for i, batch in enumerate(tqdm(dataloader)):
        hq = batch['hq'].to('cuda')
        model(hq)
        l = layer_hooked_value.clone().squeeze()
        pred = kmeans_predict(l, centers, device=l.device)
        for b in range(pred.shape[0]):
            if pred[b] == 3:
                outpath = os.path.dirname(batch['HQ_path'][b]).replace('\\pn_coven\\cropped', '\\pn_coven\\modeling')
                os.makedirs(outpath, exist_ok=True)
                shutil.move(batch['HQ_path'][b], outpath)


if __name__ == '__main__':
    pretrained_path = '../../../experiments/resnet_byol_diffframe_115k.pth'
    model = resnet50(pretrained=False).to('cuda')
    sd = torch.load(pretrained_path)
    resnet_sd = {}
    for k, v in sd.items():
        if 'target_encoder.net.' in k:
            resnet_sd[k.replace('target_encoder.net.', '')] = v
    model.load_state_dict(sd, strict=True)
    model.eval()
    register_hook(model, 'avgpool')

    with torch.no_grad():
        #find_similar_latents(model, structural_euc_dist)
        produce_latent_dict(model)
        #build_kmeans()
        #use_kmeans()
