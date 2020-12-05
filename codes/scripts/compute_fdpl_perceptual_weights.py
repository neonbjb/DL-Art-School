import torch
import numpy as np
from utils import options as option
from data import create_dataloader, create_dataset
import math
from tqdm import tqdm
from utils.fdpl_util import dct_2d, extract_patches_2d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.colors import rgb2ycbcr
import torch.nn.functional as F

input_config = "../../options/train_imgset_pixgan_srg4_fdpl.yml"
output_file = "fdpr_diff_means.pt"
device = 'cuda'
patch_size=128

if __name__ == '__main__':
    opt = option.parse(input_config, is_train=True)
    opt['dist'] = False

    # Create a dataset to load from (this dataset loads HR/LR images and performs any distortions specified by the YML.
    dataset_opt = opt['datasets']['train']
    train_set = create_dataset(dataset_opt)
    train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
    total_iters = int(opt['train']['niter'])
    total_epochs = int(math.ceil(total_iters / train_size))
    train_loader = create_dataloader(train_set, dataset_opt, opt, None)
    print('Number of train images: {:,d}, iters: {:,d}'.format(
        len(train_set), train_size))

    # calculate the perceptual weights
    master_diff = np.zeros((patch_size, patch_size))
    num_patches = 0
    all_diff_patches = []
    tq = tqdm(train_loader)
    sampled = 0
    for train_data in tq:
        if sampled > 200:
            break
        sampled += 1

        im = rgb2ycbcr(train_data['hq'].double())
        im_LR = rgb2ycbcr(F.interpolate(train_data['lq'].double(),
                                        size=im.shape[2:],
                                        mode="bicubic", align_corners=False))
        patches_hr = extract_patches_2d(img=im, patch_shape=(patch_size,patch_size), batch_first=True)
        patches_hr = dct_2d(patches_hr, norm='ortho')
        patches_lr = extract_patches_2d(img=im_LR, patch_shape=(patch_size,patch_size), batch_first=True)
        patches_lr = dct_2d(patches_lr, norm='ortho')
        b, p, c, w, h = patches_hr.shape
        diffs = torch.abs(patches_lr - patches_hr) / ((torch.abs(patches_lr) + torch.abs(patches_hr)) / 2 + .00000001)
        num_patches += b * p
        all_diff_patches.append(torch.sum(diffs, dim=(0, 1)))

    diff_patches = torch.stack(all_diff_patches, dim=0)
    diff_means = torch.sum(diff_patches, dim=0) / num_patches

    torch.save(diff_means, output_file)
    print(diff_means)

    for i in range(3):
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax.imshow(diff_means[i].numpy())
        ax.set_title("mean_diff for channel %i" % (i,))
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()

