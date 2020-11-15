"""A multi-thread tool to crop large images to sub-images for faster IO."""
import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
import data.util as data_util  # noqa: E402
import torch.utils.data as data
from tqdm import tqdm
import torch


def main():
    split_img = False
    opt = {}
    opt['n_thread'] = 5
    opt['compression_level'] = 90  # JPEG compression quality rating.
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

    opt['dest'] = 'file'
    opt['input_folder'] = ['F:\\4k6k\\datasets\\images\\div2k\\DIV2K_train_HR',
                           'F:\\4k6k\\datasets\\images\\flickr\\flickr2k\\Flickr2K_HR',
                           'F:\\4k6k\\datasets\\images\\flickr\\flickr-scrape\\filtered',
                           'F:\\4k6k\\datasets\\images\\goodeats\\hq\\new_season\\images',
                           'F:\\4k6k\datasets\\images\\youtube\\images']
    opt['save_folder'] = 'F:\\4k6k\\datasets\\images\\ge_full_1024'
    opt['imgsize'] = 1024

    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))

    extract_single(opt)


class TiledDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        input_folder = opt['input_folder']
        self.images = data_util.get_image_paths('img', input_folder)[0]

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index):
        path = self.images[index]
        basename = osp.basename(path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # Greyscale not supported.
        if len(img.shape) == 2:
            return None
        h, w, c = img.shape
        # Uncomment to filter any image that doesnt meet a threshold size.
        if min(h,w) < 1024:
            return None

        # We must convert the image into a square.
        dim = min(h, w)
        # Crop the image so that only the center is left, since this is often the most salient part of the image.
        img = img[(h - dim) // 2:dim + (h - dim) // 2, (w - dim) // 2:dim + (w - dim) // 2, :]
        img = cv2.resize(img, (self.opt['imgsize'], self.opt['imgsize']), interpolation=cv2.INTER_AREA)
        cv2.imwrite(osp.join(self.opt['save_folder'], basename + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, self.opt['compression_level']])
        return None

    def __len__(self):
        return len(self.images)


def identity(x):
    return x

def extract_single(opt):
    dataset = TiledDataset(opt)
    dataloader = data.DataLoader(dataset, num_workers=opt['n_thread'], collate_fn=identity)
    tq = tqdm(dataloader)
    for spl_imgs in tq:
        pass


if __name__ == '__main__':
    main()
