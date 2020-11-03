import os.path as osp
import logging
import shutil
import time
import argparse
from collections import OrderedDict

import os

import torchvision

import utils
import utils.options as option
import utils.util as util
from data.util import bgr2ycbcr
import models.archs.SwitchedResidualGenerator_arch as srg
from models.ExtensibleTrainer import ExtensibleTrainer
from switched_conv.switched_conv_util import save_attention_to_image, save_attention_to_image_rgb
from switched_conv.switched_conv import compute_attention_specificity
from data import create_dataset, create_dataloader
from tqdm import tqdm
import torch
import models.networks as networks

if __name__ == "__main__":
    #### options
    torch.backends.cudnn.benchmark = True
    srg_analyze = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../../options/train_psnr_approximator.yml')
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
        dataset_opt['n_workers'] = 0
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt, opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = ExtensibleTrainer(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        dst_path = "F:\\playground"
        [os.makedirs(osp.join(dst_path, str(i)), exist_ok=True) for i in range(10)]

        corruptions = ['none', 'color_quantization', 'gaussian_blur', 'motion_blur', 'smooth_blur', 'noise',
                       'jpeg-medium', 'jpeg-broad', 'jpeg-normal', 'saturation', 'lq_resampling',
                       'lq_resampling4x']
        c_counter = 0
        test_set.corruptor.num_corrupts = 0
        test_set.corruptor.random_corruptions = []
        test_set.corruptor.fixed_corruptions = [corruptions[0]]
        corruption_mse = [(0,0) for _ in corruptions]

        tq = tqdm(test_loader)
        batch_size = opt['datasets']['train']['batch_size']
        for data in tq:
            need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
            model.feed_data(data, need_GT=need_GT)
            model.test()
            est_psnr = torch.mean(model.eval_state['psnr_approximate'][0], dim=[1,2,3])
            for i in range(est_psnr.shape[0]):
                im_path = data['GT_path'][i]
                torchvision.utils.save_image(model.eval_state['lq'][0][i], osp.join(dst_path, str(int(est_psnr[i]*10)), osp.basename(im_path)))
                #shutil.copy(im_path, osp.join(dst_path, str(int(est_psnr[i]*10))))

            last_mse, last_ctr = corruption_mse[c_counter % len(corruptions)]
            corruption_mse[c_counter % len(corruptions)] = (last_mse + torch.sum(est_psnr).item(), last_ctr + 1)
            c_counter += 1
            test_set.corruptor.fixed_corruptions = [corruptions[c_counter % len(corruptions)]]
            if c_counter % 100 == 0:
                for i, (mse, ctr) in enumerate(corruption_mse):
                    print("%s: %f" % (corruptions[i], mse / (ctr * batch_size)))