import os.path as osp
import logging
import random
import time
import argparse
from collections import OrderedDict

import utils
import utils.options as option
import utils.util as util
from trainer.ExtensibleTrainer import ExtensibleTrainer
from data import create_dataset, create_dataloader
from tqdm import tqdm
import torch
import numpy as np


def forward_pass(model, data, output_dir, opt):
    alteration_suffix = util.opt_get(opt, ['name'], '')
    denorm_range = tuple(util.opt_get(opt, ['image_normalization_range'], [0, 1]))
    with torch.no_grad():
        model.feed_data(data, 0, need_GT=need_GT)
        model.test()

    visuals = model.get_current_visuals(need_GT)['rlt'].cpu()
    visuals = (visuals - denorm_range[0]) / (denorm_range[1]-denorm_range[0])
    fea_loss = 0
    psnr_loss = 0
    for i in range(visuals.shape[0]):
        img_path = data['GT_path'][i] if need_GT else data['LQ_path'][i]
        img_name = osp.splitext(osp.basename(img_path))[0]

        sr_img = util.tensor2img(visuals[i])  # uint8

        # save images
        suffix = alteration_suffix
        if suffix:
            save_img_path = osp.join(output_dir, img_name + suffix + '.png')
        else:
            save_img_path = osp.join(output_dir, img_name + '.png')

        if need_GT:
            psnr_sr = util.tensor2img(visuals[i])
            psnr_gt = util.tensor2img(data['hq'][i])
            psnr_loss += util.calculate_psnr(psnr_sr, psnr_gt)

        util.save_img(sr_img, save_img_path)
    return fea_loss, psnr_loss


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

    #### Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = ExtensibleTrainer(opt)
    fea_loss = 0
    psnr_loss = 0
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []

        tq = tqdm(test_loader)
        for data in tq:
            need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
            need_GT = need_GT and want_metrics

            fea_loss, psnr_loss = forward_pass(model, data, dataset_dir, opt)
            fea_loss += fea_loss
            psnr_loss += psnr_loss

        # log
        logger.info('# Validation # Fea: {:.4e}, PSNR: {:.4e}'.format(fea_loss / len(test_loader), psnr_loss / len(test_loader)))
