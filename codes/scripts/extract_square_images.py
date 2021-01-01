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
    opt['n_thread'] = 4
    opt['compression_level'] = 90  # JPEG compression quality rating.
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

    opt['dest'] = 'file'
    opt['input_folder'] = ['F:\\4k6k\\datasets\\ns_images\\vixen\\vix_cropped']
    opt['save_folder'] = 'F:\\4k6k\\datasets\\ns_images\\video_512_cropped'
    opt['imgsize'] = 512
    #opt['bottom_crop'] = 120

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
        print("Found %i images" % (len(self.images),))

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index):
        path = self.images[index]
        basename = osp.basename(path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # Greyscale not supported.
        if img is None:
            print("Error with ", path)
            return None
        if len(img.shape) == 2:
            print("Skipping due to greyscale")
            return None

        # Perform explicit crops first. These are generally used to get rid of watermarks so we dont even want to
        # consider these areas of the image.
        if 'bottom_crop' in self.opt.keys():
            img = img[:-self.opt['bottom_crop'], :, :]

        h, w, c = img.shape
        # Uncomment to filter any image that doesnt meet a threshold size.
        if min(h,w) < 512:
            print("Skipping due to threshold")
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
