import os
import os.path as osp
import logging
import random
import time
import argparse
from collections import OrderedDict

import numpy
from PIL import Image
from torchvision.transforms import ToTensor

import utils
import utils.options as option
import utils.util as util
from trainer.ExtensibleTrainer import ExtensibleTrainer
from data import create_dataset, create_dataloader
from tqdm import tqdm
import torch
import numpy as np

# A rough copy of test.py that "surfs" along a spectrum of timestep spacings to show what differences can be gleaned.

def forward_pass(model, data, output_dir, spacing):
    with torch.no_grad():
        model.feed_data(data, 0)
        model.test()

    visuals = model.get_current_visuals()['rlt'].cpu()
    img_path = data['GT_path'][0]
    img_name = osp.splitext(osp.basename(img_path))[0]
    sr_img = util.tensor2img(visuals[0])  # uint8

    # save images
    suffixes = [f'_{int(spacing)}']
    for suffix in suffixes:
        save_img_path = osp.join(output_dir, img_name + suffix + '.png')
        util.save_img(sr_img, save_img_path)


if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(5555)
    random.seed(5555)
    np.random.seed(5555)

    #### options
    torch.backends.cudnn.benchmark = True
    want_metrics = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../options/test_diffusion_unet.yml')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    utils.util.loaded_options = opt

    util.mkdirs(
        (path for key, path in opt['path'].items()
         if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    # Load test image
    im = ToTensor()(Image.open(opt['image'])) * 2 - 1
    _, h, w = im.shape
    if h % 2 == 1:
        im = im[:,1:,:]
        h = h-1
    if w % 2 == 1:
        im = im[:,:,1:]
        w = w-1
    dh, dw = (h - 32 * (h // 32)) // 2, (w - 32 * (w // 32)) // 2
    if dh > 0:
        im = im[:,dh:-dh]
    if dw > 0:
        im = im[:,:,dw:-dw]
    im = im.unsqueeze(0)

    # Build the corruption indexes we are going to use.
    timestep_spacings = opt['timestep_spacings']
    correction_factors = opt['correction_factor']

    results_dir = osp.join(opt['path']['results_root'], os.path.basename(opt['image']))
    util.mkdir(results_dir)
    for spacing in timestep_spacings:
        opt['steps']['generator']['injectors']['visual_debug']['respaced_timestep_spacing'] = spacing
        model = ExtensibleTrainer(opt)
        data = {
            'hq': im.to('cuda'),
            'corruption_entropy': torch.tensor([correction_factors], device='cuda',
                                               dtype=torch.float),
            'GT_path': opt['image']
        }
        forward_pass(model, data, results_dir, spacing)
