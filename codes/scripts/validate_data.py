# This script iterates through all the data with no worker threads and performs whatever transformations are prescribed.
# The idea is to find bad/corrupt images.

import math
import argparse
import random
import torch
from utils import util, options as option
from data import create_dataloader, create_dataset
from tqdm import tqdm
from skimage import io

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.', default='../../options/train_prog_mi1_rrdb_6bypass.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    opt['dist'] = False
    rank = -1

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            dataset_opt['n_workers'] = 0  # Force num_workers=0 to make dataloader work in process.
            train_loader = create_dataloader(train_set, dataset_opt, opt, None)
            if rank <= 0:
                print('Number of training data elements: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
    assert train_loader is not None

    '''
    tq_ldr = tqdm(train_set.get_paths())
    for path in tq_ldr:
        try:
            _ = io.imread(path)
            # Do stuff with img
        except Exception as e:
            print("Error with %s" % (path,))
            print(e)
    '''
    tq_ldr = tqdm(train_set)
    for ds in tq_ldr:
        pass


if __name__ == '__main__':
    main()
