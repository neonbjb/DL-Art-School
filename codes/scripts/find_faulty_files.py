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

current_batch = None
output_file = open('find_faulty_files_results.tsv', 'a')

class LossWrapper:
    def __init__(self, lwrap):
        self.lwrap = lwrap
        self.opt = lwrap.opt

    def is_stateful(self):
        return self.lwrap.is_stateful()

    def extra_metrics(self):
        return self.lwrap.extra_metrics()

    def clear_metrics(self):
        self.lwrap.clear_metrics()

    def __call__(self, m, state):
        global current_batch
        global output_file
        val = state[self.lwrap.key]
        assert val.shape[0] == len(current_batch['path'])
        val = val.view(val.shape[0], -1)
        val = val.mean(dim=1)
        errant = torch.nonzero(val > 8)
        for i in errant:
            print(f"ERRANT FOUND: {val[i]} path: {current_batch['path'][i]}")
            output_file.write(current_batch['path'][i] + "\n")
        output_file.flush()
        return self.lwrap(m, state)


# Script that builds an ExtensibleTrainer, then a pertinent loss with the above LossWrapper. The
# LossWrapper then croaks when it finds an input that produces a divergent loss
if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(5555)
    random.seed(5555)
    np.random.seed(5555)

    #### options
    torch.backends.cudnn.benchmark = True
    want_metrics = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../experiments/clean_with_lrdvae.yml')
    opt = option.parse(parser.parse_args().opt, is_train=True)
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
    dataset = create_dataset(opt['datasets']['train'])
    dataloader = create_dataloader(dataset, opt['datasets']['train'])
    logger.info('Number of test images in [{:s}]: {:d}'.format(opt['datasets']['train']['name'], len(dataset)))

    model = ExtensibleTrainer(opt)
    assert len(model.steps) == 1

    step = model.steps[0]
    step.losses['reconstruction_loss'] = LossWrapper(step.losses['reconstruction_loss'])

    for i, data in enumerate(tqdm(dataloader)):
        current_batch = data
        model.feed_data(data, i)
        model.optimize_parameters(i, optimize=False)


