import os
import shutil

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize
from tqdm import tqdm
import numpy as np

from data.image_folder_dataset import ImageFolderDataset
from models.archs.spinenet_arch import SpineNet


# Computes the structural euclidean distance between [x,y]. "Structural" here means the [h,w] dimensions are preserved
# and the distance is computed across the channel dimension.
def structural_euc_dist(x, y):
    diff = torch.square(x - y)
    sum = torch.sum(diff, dim=-1)
    return torch.sqrt(sum)


def cosine_similarity(x, y):
    x = norm(x)
    y = norm(y)
    return -nn.CosineSimilarity()(x, y)   # probably better to just use this class to perform the calc. Just left this here to remind myself.


def norm(x):
    sh = x.shape
    sh_r = tuple([sh[i] if i != len(sh)-1 else 1 for i in range(len(sh))])
    return (x - torch.mean(x, dim=-1).reshape(sh_r)) / torch.std(x, dim=-1).reshape(sh_r)


def im_norm(x):
    return (((x - torch.mean(x, dim=(2,3)).reshape(-1,1,1,1)) / torch.std(x, dim=(2,3)).reshape(-1,1,1,1)) * .5) + .5


def get_image_folder_dataloader(batch_size, num_workers):
    dataset_opt = {
        'name': 'amalgam',
        #'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\imageset_1024_square_with_new'],
        'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\1024_test'],
        'weights': [1],
        'target_size': 512,
        'force_multiple': 32,
        'scale': 1
    }
    dataset = ImageFolderDataset(dataset_opt)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def create_latent_database(model):
    batch_size = 8
    num_workers = 1
    output_path = '../../results/byol_spinenet_latents/'

    os.makedirs(output_path, exist_ok=True)
    dataloader = get_image_folder_dataloader(batch_size, num_workers)
    id = 0
    dict_count = 1
    latent_dict = {}
    all_paths = []
    for batch in tqdm(dataloader):
        hq = batch['hq'].to('cuda:1')
        latent = model(hq)[1]   # BYOL trainer only trains the '4' output, which is indexed at [1]. Confusing.
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


def _get_mins_from_latent_dictionary(latent, hq_img_repo, ld_file_name, batch_size):
    _, c, h, w = latent.shape
    lat_dict = torch.load(os.path.join(hq_img_repo, ld_file_name))
    comparables = torch.stack(list(lat_dict.values()), dim=0).permute(0,2,3,1)
    cbl_shape = comparables.shape[:3]
    assert cbl_shape[1] == 32
    comparables = comparables.reshape(-1, c)

    clat = latent.reshape(1,-1,h*w).permute(2,0,1)
    cpbl_chunked = torch.chunk(comparables, len(comparables) // batch_size)
    assert len(comparables) % batch_size == 0   # The reconstruction logic doesn't work if this is not the case.
    mins = []
    min_offsets = []
    for cpbl_chunk in tqdm(cpbl_chunked):
        cpbl_chunk = cpbl_chunk.to('cuda:1')
        dist = structural_euc_dist(clat, cpbl_chunk.unsqueeze(0))
        _min = torch.min(dist, dim=-1)
        mins.append(_min[0])
        min_offsets.append(_min[1])
    mins = torch.min(torch.stack(mins, dim=-1), dim=-1)
    # There's some way to do this in torch, I just can't figure it out..
    for i in range(len(mins[1])):
        mins[1][i] = mins[1][i] * batch_size + min_offsets[mins[1][i]][i]

    return mins[0].cpu(), mins[1].cpu(), len(comparables)


def find_similar_latents(model):
    img = 'F:\\4k6k\\datasets\\ns_images\\adrianna\\analyze\\analyze_xx\\adrianna_xx.jpg'
    #img = 'F:\\4k6k\\datasets\\ns_images\\adrianna\\analyze\\analyze_xx\\nicky_xx.jpg'
    hq_img_repo = '../../results/byol_spinenet_latents'
    output_path = '../../results/byol_spinenet_similars'
    batch_size = 1024
    num_maps = 8

    os.makedirs(output_path, exist_ok=True)
    img_bank_paths = torch.load(os.path.join(hq_img_repo, "all_paths.pth"))
    img_t = ToTensor()(Image.open(img)).to('cuda:1').unsqueeze(0)
    _, _, h, w = img_t.shape
    img_t = img_t[:, :, :128*(h//128), :128*(w//128)]

    latent = model(img_t)[1]
    _, c, h, w = latent.shape
    mins, min_offsets = [], []
    total_latents = -1
    for d_id in range(1,num_maps+1):
        mn, of, tl = _get_mins_from_latent_dictionary(latent, hq_img_repo, "latent_dict_%i.pth" % (d_id), batch_size)
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
    <html><body><img id="imgmap" src="source.png" usemap="#map">
    <map name="map">%s</map><br>
    <button onclick="if(imgmap.src.includes('output.png')){imgmap.src='source.png';}else{imgmap.src='output.png';}">Swap Images</button>
    </body></html>
    '''
    img_map_areas = []
    img_out = torch.zeros((1,3,h*16,w*16))
    for i, ind in enumerate(tqdm(min_ids)):
        u = np.unravel_index(ind.item(), (num_maps*total_latents//(32*32),32,32))
        h_, w_ = np.unravel_index(i, (h, w))

        img = ToTensor()(Resize((512, 512))(Image.open(img_bank_paths[u[0]])))
        t = 16 * u[1]
        l = 16 * u[2]
        patch = img[:, t:t+16, l:l+16]
        img_out[:,:,h_*16:h_*16+16,w_*16:w_*16+16] = patch

        # Also save the image with a masked map
        mask = torch.full_like(img, fill_value=.3)
        mask[:, t:t+16, l:l+16] = 1
        masked_img = img * mask
        masked_src_img_output_file = os.path.join(output_path, "%i_%i__%i.png" % (t, l, u[0]))
        torchvision.utils.save_image(masked_img, masked_src_img_output_file)

        # Update the image map areas.
        img_map_areas.append('<area shape="rect" coords="%i,%i,%i,%i" href="%s">' % (w_*16,h_*16,w_*16+16,h_*16+16,masked_src_img_output_file))
    torchvision.utils.save_image(img_out, os.path.join(output_path, "output.png"))
    torchvision.utils.save_image(img_t, os.path.join(output_path, "source.png"))
    doc_out = doc_out % ('\n'.join(img_map_areas))
    with open(os.path.join(output_path, 'map.html'), 'w') as f:
        print(doc_out, file=f)


def explore_latent_results(model):
    batch_size = 16
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
        dist = cosine_similarity(latent, centers).unsqueeze(1)
        dist = im_norm(dist)
        torchvision.utils.save_image(dist, os.path.join(output_path, "%i.png" % id))
        id += 1


if __name__ == '__main__':
    pretrained_path = '../../experiments/spinenet49_imgset_byol.pth'

    model = SpineNet('49', in_channels=3, use_input_norm=True).to('cuda:1')
    model.load_state_dict(torch.load(pretrained_path), strict=True)
    model.eval()

    with torch.no_grad():
        find_similar_latents(model)