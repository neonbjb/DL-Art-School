import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize
from tqdm import tqdm
import numpy as np

from data.images.image_folder_dataset import ImageFolderDataset
from models.image_latents.spinenet_arch import SpineNet


# Computes the structural euclidean distance between [x,y]. "Structural" here means the [h,w] dimensions are preserved
# and the distance is computed across the channel dimension.
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
        'weights': [1],
        'target_size': 256,
        'force_multiple': 32,
        'scale': 1
    })
    dataset = ImageFolderDataset(dataset_opt)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def create_latent_database(model, model_index=0, batch_size=8):
    num_workers = 4
    output_path = '../results/byol_latents/'

    os.makedirs(output_path, exist_ok=True)
    dataloader = get_image_folder_dataloader(batch_size, num_workers)
    id = 0
    dict_count = 1
    latent_dict = {}
    all_paths = []
    for batch in tqdm(dataloader):
        hq = batch['hq'].to('cuda')
        latent = model(hq)
        if isinstance(latent, tuple):
            latent = latent[model_index]
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


def _get_mins_from_comparables(latent, comparables, batch_size, compare_fn):
    _, c, h, w = latent.shape
    clat = latent.reshape(1,-1,h*w).permute(2,0,1)
    cpbl_chunked = torch.chunk(comparables, len(comparables) // batch_size)
    assert len(comparables) % batch_size == 0   # The reconstruction logic doesn't work if this is not the case.
    mins = []
    min_offsets = []
    for cpbl_chunk in tqdm(cpbl_chunked):
        cpbl_chunk = cpbl_chunk.to('cuda')
        dist = compare_fn(clat, cpbl_chunk.unsqueeze(0))
        _min = torch.min(dist, dim=-1)
        mins.append(_min[0])
        min_offsets.append(_min[1])
    mins = torch.min(torch.stack(mins, dim=-1), dim=-1)

    # There's some way to do this in torch, I just can't figure it out..
    for i in range(len(mins[1])):
        mins[1][i] = mins[1][i] * batch_size + min_offsets[mins[1][i]][i]

    return mins[0].cpu(), mins[1].cpu(), len(comparables)


def _get_mins_from_latent_dictionary(latent, hq_img_repo, ld_file_name, batch_size, compare_fn):
    _, c, h, w = latent.shape
    lat_dict = torch.load(os.path.join(hq_img_repo, ld_file_name))
    comparables = torch.stack(list(lat_dict.values()), dim=0).permute(0,2,3,1)
    cbl_shape = comparables.shape[:3]
    comparables = comparables.reshape(-1, c)
    return _get_mins_from_comparables(latent, comparables, batch_size, compare_fn)


def find_similar_latents(model, model_index=0, lat_patch_size=16, compare_fn=structural_euc_dist):
    img = 'F:\\4k6k\\datasets\\ns_images\\adrianna\\analyze\\analyze_xx\\adrianna_xx.jpg'
    #img = 'F:\\4k6k\\datasets\\ns_images\\adrianna\\analyze\\analyze_xx\\nicky_xx.jpg'
    hq_img_repo = '../results/byol_latents'
    output_path = '../results/byol_similars'
    batch_size = 4096
    num_maps = 1
    lat_patch_mult = 512 // lat_patch_size

    os.makedirs(output_path, exist_ok=True)
    img_bank_paths = torch.load(os.path.join(hq_img_repo, "all_paths.pth"))
    img_t = ToTensor()(Image.open(img)).to('cuda').unsqueeze(0)
    _, _, h, w = img_t.shape
    img_t = img_t[:, :, :128*(h//128), :128*(w//128)]
    latent = model(img_t)
    if not isinstance(latent, tuple):
        latent = (latent,)
    latent = latent[model_index]
    _, c, h, w = latent.shape

    mins, min_offsets = [], []
    total_latents = -1
    for d_id in range(1,num_maps+1):
        mn, of, tl = _get_mins_from_latent_dictionary(latent, hq_img_repo, "latent_dict_%i.pth" % (d_id), batch_size, compare_fn)
        if total_latents != -1:
            assert total_latents == tl
        else:
            total_latents = tl
        mins.append(mn)
        min_offsets.append(of)
    mins = torch.min(torch.stack(mins, dim=-1), dim=-1)
    # There's some way to do this in torch, I just can't figure it out..
    for i in range(len(mins[1])):
        mins[1][i] = mins[1][i] * total_latents + min_offsets[mins[1][i]][i]
    min_ids = mins[1]

    print("Constructing image map..")
    doc_out = '''
    <html><body><img id="imgmap" src="output.png" usemap="#map">
    <map name="map">%s</map><br>
    <button onclick="if(imgmap.src.includes('output.png')){imgmap.src='source.png';}else{imgmap.src='output.png';}">Swap Images</button>
    </body></html>
    '''
    img_map_areas = []
    img_out = torch.zeros((1, 3, h * lat_patch_size, w * lat_patch_size))
    for i, ind in enumerate(tqdm(min_ids)):
        u = np.unravel_index(ind.item(), (num_maps * total_latents // (lat_patch_mult ** 2), lat_patch_mult, lat_patch_mult))
        h_, w_ = np.unravel_index(i, (h, w))

        img = ToTensor()(Resize((512, 512))(Image.open(img_bank_paths[u[0]])))
        t = lat_patch_size * u[1]
        l = lat_patch_size * u[2]
        patch = img[:, t:t + lat_patch_size, l:l + lat_patch_size]
        io_loc_t = h_ * lat_patch_size
        io_loc_l = w_ * lat_patch_size
        img_out[:,:,io_loc_t:io_loc_t+lat_patch_size,io_loc_l:io_loc_l+lat_patch_size] = patch

        # Also save the image with a masked map
        mask = torch.full_like(img, fill_value=.3)
        mask[:, t:t + lat_patch_size, l:l + lat_patch_size] = 1
        masked_img = img * mask
        masked_src_img_output_file = os.path.join(output_path, "%i_%i__%i.png" % (io_loc_t, io_loc_l, u[0]))
        torchvision.utils.save_image(masked_img, masked_src_img_output_file)

        # Update the image map areas.
        img_map_areas.append('<area shape="rect" coords="%i,%i,%i,%i" href="%s">' % (io_loc_l, io_loc_t,
                                                                                     io_loc_l + lat_patch_size, io_loc_t + lat_patch_size,
                                                                                     masked_src_img_output_file))
    torchvision.utils.save_image(img_out, os.path.join(output_path, "output.png"))
    torchvision.utils.save_image(img_t, os.path.join(output_path, "source.png"))
    doc_out = doc_out % ('\n'.join(img_map_areas))
    with open(os.path.join(output_path, 'map.html'), 'w') as f:
        print(doc_out, file=f)


def explore_latent_results(model):
    batch_size = 16
    num_workers = 4
    output_path = '../../results/byol_spinenet_explore_latents/'

    os.makedirs(output_path, exist_ok=True)
    dataloader = get_image_folder_dataloader(batch_size, num_workers)
    id = 0
    for batch in tqdm(dataloader):
        hq = batch['hq'].to('cuda')
        latent = model(hq)[1]   # BYOL trainer only trains the '4' output, which is indexed at [1]. Confusing.
        # This operation works by computing the distance of every structural index from the center and using that
        # as a "heatmap".
        b, c, h, w = latent.shape
        center = latent[:, :, h//2, w//2].unsqueeze(-1).unsqueeze(-1)
        centers = center.repeat(1, 1, h, w)
        dist = cosine_similarity(latent, centers).unsqueeze(1)
        dist = im_norm(dist)
        torchvision.utils.save_image(dist, os.path.join(output_path, "%i.png" % id))
        id += 1


class BYOLModelWrapper(nn.Module):
    def __init__(self, wrap):
        super().__init__()
        self.wrap = wrap

    def forward(self, img):
        return self.wrap.get_projection(img)


if __name__ == '__main__':
    pretrained_path = '../../../experiments/spinenet49_imgset_sbyol.pth'
    model = SpineNet('49', in_channels=3, use_input_norm=True).to('cuda')
    model.load_state_dict(torch.load(pretrained_path), strict=True)
    model.eval()

    #util.loaded_options = {'checkpointing_enabled': True}
    #pretrained_path = '../../experiments/train_sbyol_512unsupervised_restart/models/48000_generator.pth'
    #from models.byol.byol_structural import StructuralBYOL
    #subnet = SpineNet('49', in_channels=3, use_input_norm=True).to('cuda')
    #model = StructuralBYOL(subnet, image_size=256, hidden_layer='endpoint_convs.4.conv')
    #model.load_state_dict(torch.load(pretrained_path), strict=True)
    #model = BYOLModelWrapper(model)
    #model.eval()

    with torch.no_grad():
        #create_latent_database(model, 1)    # 0 = model output dimension to use for latent storage
        find_similar_latents(model, 1, 16, structural_euc_dist)  # 1 = model output dimension to use for latent predictor.
