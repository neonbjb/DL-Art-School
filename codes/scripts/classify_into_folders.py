import os.path as osp
import logging
import shutil
import time
import argparse

import os

import utils
import utils.options as option
import utils.util as util
from trainer.ExtensibleTrainer import ExtensibleTrainer
from data import create_dataset, create_dataloader
from tqdm import tqdm
import torch
import torchvision


if __name__ == "__main__":
    #### options
    torch.backends.cudnn.benchmark = True
    want_metrics = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../options/test_noisy_audio_clips_classifier.yml')
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
        if 'test' in phase:
            test_set = create_dataset(dataset_opt)
            if hasattr(test_set, 'wrapped_dataset'):
                test_set = test_set.wrapped_dataset
            test_loader = create_dataloader(test_set, dataset_opt, opt)
            logger.info('Number of test images: {:d}'.format(len(test_set)))
            test_loaders.append((dataset_opt['name'], test_loader))

    model = ExtensibleTrainer(opt)
    # Remove all losses, since often labels will not be provided and this is not implied in test.
    for s in model.steps:
        s.losses = {}

    output_key = opt['eval']['classifier_logits_key']
    labels = opt['eval']['output_labels']
    path_key = opt['eval']['path_key']
    output_base_dir = util.opt_get(opt, ['eval', 'output_dir'], None)
    output_file = open('classify_into_folders.tsv', 'a')

    step = 0
    for test_set_name, test_loader in test_loaders:
        test_start_time = time.time()
        dataset_dir = osp.join(opt['path']['results_root'], opt['name'])
        util.mkdir(dataset_dir)

        for data in tqdm(test_loader):
            with torch.no_grad():
                model.feed_data(data, 0)
                model.test()

            lbls = torch.nn.functional.softmax(model.eval_state[output_key][0].cpu(), dim=-1)
            for k, lbl in enumerate(lbls):
                lbl = labels[torch.argmax(lbl, dim=0)]
                src_path = data[path_key][k]
                output_file.write(f'{src_path}\t{lbl}\n')
                if output_base_dir is not None:
                    dest = os.path.join(output_base_dir, lbl)
                    os.makedirs(dest, exist_ok=True)
                    shutil.copy(str(src_path), os.path.join(dest, f'{step}_{os.path.basename(str(src_path))}'))
                    step += 1
            output_file.flush()
    output_file.close()

