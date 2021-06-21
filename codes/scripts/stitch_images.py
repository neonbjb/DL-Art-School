import glob

import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    imfolder = 'F:\\dlas\\results\\test_diffusion_unet\\imgset5'
    cols, rows = 10, 5
    images = glob.glob(f'{imfolder}/*.png')
    output = None
    for r in range(rows):
        for c in range(cols):
            im = ToTensor()(Image.open(next(images)))
            if output is None:
                c, h, w = im.shape
                output = torch.zeros(c, h * rows, w * cols)
            output[:,r*h:(r+1)*h,c*w:(c+1)*w] = im
    torchvision.utils.save_image(output, "out.png")