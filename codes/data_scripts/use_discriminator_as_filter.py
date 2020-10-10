import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import os
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
import models.archs.SwitchedResidualGenerator_arch as srg
from switched_conv.switched_conv_util import save_attention_to_image, save_attention_to_image_rgb
from switched_conv.switched_conv import compute_attention_specificity
from data import create_dataset, create_dataloader
from models import create_model
from tqdm import tqdm
import torch
import models.networks as networks
import shutil
import torchvision


if __name__ == "__main__":
    bin_path = "f:\\binned"
    good_path = "f:\\good"
    os.makedirs(bin_path, exist_ok=True)
    os.makedirs(good_path, exist_ok=True)


    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../../options/discriminator_filter.yml')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    opt['dist'] = False

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
        test_loader = create_dataloader(test_set, dataset_opt, opt=opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = create_model(opt)
    fea_loss = 0
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        tq = tqdm(test_loader)
        removed = 0
        means = []
        dataset_mean = -7.133
        for data in tq:
            model.feed_data(data, need_GT=True)
            model.test()
            results = model.eval_state['discriminator_out'][0]
            means.append(torch.mean(results).item())
            print(sum(means)/len(means), torch.mean(results), torch.max(results), torch.min(results))
            for i in range(results.shape[0]):
                #if results[i] < .8:
                #    os.remove(data['GT_path'][i])
                #    removed += 1
                imname = osp.basename(data['GT_path'][i])
                if results[i]-dataset_mean > 1:
                    torchvision.utils.save_image(data['GT'][i], osp.join(bin_path, imname))

        print("Removed %i/%i images" % (removed, len(test_set)))