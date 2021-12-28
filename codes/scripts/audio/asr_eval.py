import os
import os.path as osp
import logging
import random
import argparse

import torchvision

import utils
import utils.options as option
import utils.util as util
from models.tacotron2.text import sequence_to_text
from trainer.ExtensibleTrainer import ExtensibleTrainer
from data import create_dataset, create_dataloader
from tqdm import tqdm
import torch
import numpy as np
from scipy.io import wavfile


def forward_pass(model, data, output_dir, opt, macro_b, dataset):
    with torch.no_grad():
        model.feed_data(data, 0)
        model.test()

    gt_key = opt['eval']['gen_text']
    txts = []
    for b in range(model.eval_state[gt_key][0].shape[0]):
        if 'real_text' in opt['eval'].keys():
            real = data[opt['eval']['real_text']][b]
            print(f'{macro_b} {b} Real text: "{real}"')

        codes = model.eval_state[opt['eval']['gen_text']][0][b].cpu()
        if hasattr(dataset, 'tokenizer'):
            text = dataset.tokenizer.decode(codes.numpy())
            text = text.replace(' $$$', '')
            txts.append(text)
        else:
            txts.append(sequence_to_text(codes))
    return txts


if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(5555)
    random.seed(5555)
    np.random.seed(5555)

    #### options
    torch.backends.cudnn.benchmark = True
    want_metrics = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../options/test_gpt_asr_hf2.yml')
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

    dataset_opt = opt['datasets']['val']
    test_set, collate_fn = create_dataset(dataset_opt, return_collate=True)
    test_loader = create_dataloader(test_set, dataset_opt, collate_fn=collate_fn)
    logger.info('Number of test texts in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))

    model = ExtensibleTrainer(opt)

    batch = 0
    output = open('results.tsv', 'w')
    dataset_dir = opt['path']['results_root']
    util.mkdir(dataset_dir)

    for data in tqdm(test_loader):
        #if data['clip'].shape[-1] > opt['networks']['asr_gen']['kwargs']['max_mel_frames']*255:
        #    continue
        preds = forward_pass(model, data, dataset_dir, opt, batch, test_set)
        for b, pred in enumerate(preds):
            pred = pred.replace('_', '')
            output.write(f'{pred}\t{os.path.basename(data["filenames"][b])}\n')
            print(pred)
            batch += 1
        output.flush()

