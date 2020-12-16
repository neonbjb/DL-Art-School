import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import os

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
import torchvision


if __name__ == "__main__":
    #### options
    torch.backends.cudnn.benchmark = True
    want_metrics = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../options/train_imgset_structural_classifier.yml')
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
        dataset_opt['dataset']['includes_labels'] = False
        del dataset_opt['dataset']['labeler']
        test_set = create_dataset(dataset_opt)
        if hasattr(test_set, 'wrapped_dataset'):
            test_set = test_set.wrapped_dataset
        test_loader = create_dataloader(test_set, dataset_opt, opt)
        logger.info('Number of test images: {:d}'.format(len(test_set)))
        test_loaders.append(test_loader)

    model = ExtensibleTrainer(opt)
    gen = model.netsG['generator']
    label_to_search_for = 4

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        test_start_time = time.time()
        dataset_dir = osp.join(opt['path']['results_root'], opt['name'])
        util.mkdir(dataset_dir)

        tq = tqdm(test_loader)
        step = 1
        for data in tq:
            hq = data['hq'].to('cuda')
            res = gen(hq)
            res = torch.nn.functional.interpolate(res, size=hq.shape[2:], mode="nearest")
            res_lbl = res[:, label_to_search_for, :, :].unsqueeze(1)
            res_lbl_mask = (1.0 * (res_lbl > .5))*.5 + .5
            hq = hq * res_lbl_mask
            torchvision.utils.save_image(hq, os.path.join(dataset_dir, "%i.png" % (step,)))
            step += 1
