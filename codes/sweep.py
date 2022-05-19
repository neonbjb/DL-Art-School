import copy
import functools
import os
from multiprocessing.pool import ThreadPool

import torch

from train import Trainer
from utils import options as option
import collections.abc


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def launch_trainer(opt, opt_path, rank):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    print('export CUDA_VISIBLE_DEVICES=' + str(rank))
    trainer = Trainer()
    opt['dist'] = False
    trainer.rank = -1
    trainer.init(opt_path, opt, 'none')
    trainer.do_training()


if __name__ == '__main__':
    """
    Ad-hoc script (hard coded; no command-line parameters) that spawns multiple separate trainers from a single options
    file, with a hard-coded set of modifications.
    """
    base_opt = '../experiments/sweep_music_mel2vec.yml'
    modifications = {
        'baseline': {},
        'lr1e3': {'steps': {'generator': {'optimizer_params': {'lr': {.001}}}}},
        'lr1e5': {'steps': {'generator': {'optimizer_params': {'lr': {.00001}}}}},
        'no_warmup': {'train': {'warmup_steps': 0}},
    }
    base_rank = 4
    opt = option.parse(base_opt, is_train=True)
    all_opts = []
    for i, (mod, mod_dict) in enumerate(modifications.items()):
        nd = copy.deepcopy(opt)
        deep_update(nd, mod_dict)
        nd['name'] = f'{nd["name"]}_{mod}'
        nd['wandb_run_name'] = mod
        base_path = nd['path']['log']
        for k, p in nd['path'].items():
            if isinstance(p, str) and base_path in p:
                nd['path'][k] = p.replace(base_path, f'{base_path}/{mod}')
        all_opts.append(nd)

    for i in range(1,len(modifications)):
        pid = os.fork()
        if pid == 0:
            rank = i
            break
        else:
            rank = 0
    launch_trainer(all_opts[rank], base_opt, rank+base_rank)
