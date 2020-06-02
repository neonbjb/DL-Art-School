import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from tqdm import tqdm
import torch

if __name__ == "__main__":
    #### options
    torch.backends.cudnn.benchmark = True
    want_just_images = True
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YMAL file.', default='../options/test_resgen_upsample.yml')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

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

    model = create_model(opt)
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
            model.feed_data(data, need_GT=need_GT)
            model.test()

            if isinstance(model.fake_H, tuple):
                visuals = model.fake_H[0].detach().float().cpu()
            else:
                visuals = model.fake_H.detach().float().cpu()
            for i in range(visuals.shape[0]):
                img_path = data['GT_path'][i] if need_GT else data['LQ_path'][i]
                img_name = osp.splitext(osp.basename(img_path))[0]

                sr_img = util.tensor2img(visuals[i])  # uint8

                # save images
                suffix = opt['suffix']
                if suffix:
                    save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
                else:
                    save_img_path = osp.join(dataset_dir, img_name + '.png')
                util.save_img(sr_img, save_img_path)

                if want_just_images:
                    continue

        if not want_just_images and need_GT:  # metrics
            # Average PSNR/SSIM results
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            logger.info(
                '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
                    test_set_name, ave_psnr, ave_ssim))
            if test_results['psnr_y'] and test_results['ssim_y']:
                ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                logger.info(
                    '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.
                    format(ave_psnr_y, ave_ssim_y))
