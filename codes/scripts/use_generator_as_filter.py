import os
import shutil

from torch.utils.data import DataLoader

from data.images.single_image_dataset import SingleImageDataset
from tqdm import tqdm
import torch

from models.vqvae.vqvae_no_conv_transpose import VQVAE

if __name__ == "__main__":
    bin_path = "f:\\binned"
    good_path = "f:\\good"
    os.makedirs(bin_path, exist_ok=True)
    os.makedirs(good_path, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    model = VQVAE().cuda()
    model.load_state_dict(torch.load('../experiments/nvqvae_imgset.pth'))
    ds = SingleImageDataset({
        'name': 'amalgam',
        'paths': ['F:\\4k6k\\datasets\\ns_images\\imagesets\\256_with_ref_v5'],
        'weights': [1],
        'target_size': 128,
        'force_multiple': 32,
        'scale': 1,
        'eval': False
    })
    dl = DataLoader(ds, batch_size=256, num_workers=1)

    means = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(dl)):
            hq = data['hq'].cuda()
            gen = model(hq)[0]
            l2 = torch.mean(torch.square(hq - gen), dim=[1,2,3])
            for b in range(len(l2)):
                if l2[b] > .0004:
                    shutil.copy(data['GT_path'][b], good_path)
                #else:
                #    shutil.copy(data['GT_path'][b], bin_path)


            #means.append(l2.cpu())
            #if i % 10 == 0:
            #    print(torch.stack(means, dim=0).mean())
