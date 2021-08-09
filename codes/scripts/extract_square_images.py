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

    opt['dest'] = 'file'
    opt['input_folder'] = ['E:\\4k6k\datasets\\ns_images\\imagesets\\imageset_256_masked']
    opt['save_folder'] = 'E:\\4k6k\datasets\\ns_images\\imagesets\\imageset_128_masked'
    opt['imgsize'] = (128,128)
    opt['bottom_crop'] = 0
    opt['keep_folder'] = False

    #opt['format'] = 'jpg'
    #opt['cv2_write_options'] = [cv2.IMWRITE_JPEG_QUALITY, 95]
    opt['format'] = 'png'
    opt['cv2_write_options'] = [cv2.IMWRITE_PNG_COMPRESSION, 9]

    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))

    extract_single(opt)


class TiledDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        input_folder = opt['input_folder']
        self.images = data_util.find_files_of_type('img', input_folder)[0]
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
        if 'bottom_crop' in self.opt.keys() and self.opt['bottom_crop'] > 0:
            bc = self.opt['bottom_crop']
            if bc > 0 and bc < 1:
                bc = int(bc * img.shape[0])
            img = img[:-bc, :, :]

        h, w, c = img.shape
        # Uncomment to filter any image that doesnt meet a threshold size.
        imgsz_w, imgsz_h = self.opt['imgsize']
        if w < imgsz_w or h < imgsz_h:
            print("Skipping due to threshold")
            return None

        # We must first center-crop the image to the proper aspect ratio
        aspect_ratio = imgsz_h / imgsz_w
        if h < w * aspect_ratio:
            hdim = h
            wdim = int(h / aspect_ratio)
        elif w * aspect_ratio < h:
            hdim = int(w * aspect_ratio)
            wdim = w
        else:
            hdim = h
            wdim = w
        img = img[(h - hdim) // 2:hdim + (h - hdim) // 2, (w - wdim) // 2:wdim + (w - wdim) // 2, :]

        img = cv2.resize(img, (imgsz_w, imgsz_h), interpolation=cv2.INTER_AREA)
        output_folder = self.opt['save_folder']
        if self.opt['keep_folder']:
            # Attempt to find the folder name one level above opt['input_folder'] and use that.
            pts = [os.path.dirname(path)]
            while pts[0] != self.opt['input_folder'][0]:
                pts = os.path.split(pts[0])
            output_folder = osp.join(self.opt['save_folder'], pts[-1])
            os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(osp.join(output_folder, basename.replace('.webp', self.opt['format'])), img, self.opt['cv2_write_options'])
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
