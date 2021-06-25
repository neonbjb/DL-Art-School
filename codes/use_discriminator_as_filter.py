import os.path as osp
import logging
import time
import argparse

import os

from torchvision.transforms import CenterCrop

from trainer.ExtensibleTrainer import ExtensibleTrainer
from utils import options as option
import utils.util as util
from data import create_dataset, create_dataloader
from tqdm import tqdm
import torch
import torchvision


if __name__ == "__main__":
    bin_path = "f:\\tmp\\binned"
    good_path = "f:\\tmp\\good"
    os.makedirs(bin_path, exist_ok=True)
    os.makedirs(good_path, exist_ok=True)


    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../options/train_quality_detectors/train_resnet_jpeg.yml')
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

    model = ExtensibleTrainer(opt)
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
        for k, data in enumerate(tq):
            model.feed_data(data, k)
            model.test()
            results = torch.argmax(torch.nn.functional.softmax(model.eval_state['logits'][0], dim=-1), dim=1)
            for i in range(results.shape[0]):
                if results[i] == 0:
                    imname = osp.basename(data['HQ_path'][i])
                    # For VERIFICATION:
                    #torchvision.utils.save_image(data['hq'][i], osp.join(bin_path, imname))
                    # 4 REALZ:
                    os.remove(data['HQ_path'][i])
                    removed += 1

        print("Removed %i/%i images" % (removed, len(test_set)))