import os.path as osp
import logging
import random
import argparse

import utils
import utils.options as option
import utils.util as util
from models.waveglow.denoiser import Denoiser
from trainer.ExtensibleTrainer import ExtensibleTrainer
from data import create_dataset, create_dataloader
from tqdm import tqdm
import torch
import numpy as np
from scipy.io import wavfile


def forward_pass(model, denoiser, data, output_dir, opt, b):
    with torch.no_grad():
        model.feed_data(data, 0)
        model.test()
    pred_waveforms = model.eval_state[opt['eval']['output_state']][0]
    pred_waveforms = denoiser(pred_waveforms)
    ground_truth_waveforms = model.eval_state[opt['eval']['ground_truth']][0]
    ground_truth_waveforms = denoiser(ground_truth_waveforms)
    for i in range(pred_waveforms.shape[0]):
        audio = pred_waveforms[i][0].cpu().numpy()
        wavfile.write(osp.join(output_dir, f'{b}_{i}.wav'), 22050, audio)
        audio = ground_truth_waveforms[i][0].cpu().numpy()
        wavfile.write(osp.join(output_dir, f'{b}_{i}_ground_truth.wav'), 22050, audio)


if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(5555)
    random.seed(5555)
    np.random.seed(5555)

    #### options
    torch.backends.cudnn.benchmark = True
    want_metrics = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../options/test_gpt_tts_lj.yml')
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

    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set, collate_fn = create_dataset(dataset_opt, return_collate=True)
        test_loader = create_dataloader(test_set, dataset_opt, collate_fn=collate_fn)
        logger.info('Number of test texts in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = ExtensibleTrainer(opt)

    denoiser = Denoiser(model.networks['waveglow'].module)  # Pretty hacky, need to figure out a better way to integrate this.

    batch = 0
    for test_loader in test_loaders:
        dataset_dir = opt['path']['results_root']
        util.mkdir(dataset_dir)

        tq = tqdm(test_loader)
        for data in tq:
            forward_pass(model, denoiser, data, dataset_dir, opt, batch)
            batch += 1

