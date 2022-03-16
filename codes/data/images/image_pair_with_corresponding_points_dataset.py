import torch
import os

import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# Builds a dataset created from a simple folder containing a list of training/test/validation images.


class ImagePairWithCorrespondingPointsDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.path = opt['path']
        self.pairs = list(filter(lambda f: not os.path.isdir(f), os.listdir(self.path)))
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                              ])
        self.size = opt['size']


    def __getitem__(self, item):
        dir = self.pairs[item]
        img1 = self.transforms(Image.open(os.path.join(self.path, dir, "1.jpg")))
        img2 = self.transforms(Image.open(os.path.join(self.path, dir, "2.jpg")))
        coords1, coords2 = torch.load(os.path.join(self.path, dir, "coords.pth"))
        assert img1.shape[-2] == img1.shape[-1]
        assert img2.shape[-2] == img2.shape[-1]
        if img1.shape[-1] != self.size:
            scale = img1.shape[-1] / self.size
            assert(int(scale) == scale)  # We will only downsample to even resolutions.
            scale = 1 / scale
            img1 = torch.nn.functional.interpolate(img1.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
            coords1 = [int(c * scale) for c in coords1]
        if img2.shape[-1] != self.size:
            scale = img2.shape[-1] / self.size
            assert(int(scale) == scale)  # We will only downsample to even resolutions.
            scale = 1 / scale
            img2 = torch.nn.functional.interpolate(img2.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
            coords2 = [int(c * scale) for c in coords2]
        coords1 = (coords1[1], coords1[0])  # The UI puts these out backwards (x,y). Flip them.
        coords2 = (coords2[1], coords2[0])
        return {
            'img1': img1,
            'img2': img2,
            'coords1': coords1,
            'coords2': coords2
        }

    def __len__(self):
        return len(self.pairs)

if __name__ == '__main__':
    opt = {
        'path': 'F:\\dlas\\codes\\scripts\\ui\\image_pair_labeler\\results',
        'size': 256
    }
    output_path = '..'

    ds = DataLoader(ImagePairWithCorrespondingPointsDataset(opt), shuffle=True, num_workers=0)
    for i, d in tqdm(enumerate(ds)):
        i1 = d['img1']
        i2 = d['img2']
        c1 = d['coords1']
        c2 = d['coords2']
        i1[:,:,c1[0]-3:c1[0]+3,c1[1]-3:c1[1]+3] = 0
        i2[:,:,c2[0]-3:c2[0]+3,c2[1]-3:c2[1]+3] = 0
        torchvision.utils.save_image(i1, f'{output_path}\\{i}_1.png')
        torchvision.utils.save_image(i2, f'{output_path}\\{i}_2.png')