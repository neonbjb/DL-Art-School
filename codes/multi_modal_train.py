# This is a wrapper around train.py which allows you to train a set of models using a variety of different training
# paradigms. This works by using the yielding mechanism built into train.py to iterate one step at a time and
# synchronize the underlying models.
#
# Note that this wrapper is **EXTREMELY** simple and doesn't attempt to do many things. Some issues you should plan for:
# 1) Each trainer will have its own optimizer for the underlying model - even when the model is shared.
# 2) Each trainer will run validation and save model states according to its own schedule. Likewise:
# 3) Each trainer will load state params for the models it controls independently, regardless of whether or not those
#    models are shared. Your best bet is to have all models save state at the same time so that they all load ~ the same
#    state when re-started.
import argparse

import yaml

import train
import utils.options as option
from utils.util import OrderedYaml
import torch


def main(master_opt, launcher):
    trainers = []
    all_networks = {}
    shared_networks = []
    if launcher != 'none':
        train.init_dist('nccl')
    for i, sub_opt in enumerate(master_opt['trainer_options']):
        sub_opt_parsed = option.parse(sub_opt, is_train=True)
        trainer = train.Trainer()

        #### distributed training settings
        if launcher == 'none':  # disabled distributed training
            sub_opt_parsed['dist'] = False
            trainer.rank = -1
            print('Disabled distributed training.')
        else:
            sub_opt_parsed['dist'] = True
            trainer.world_size = torch.distributed.get_world_size()
            trainer.rank = torch.distributed.get_rank()

        trainer.init(sub_opt_parsed, launcher, all_networks)
        train_gen = trainer.create_training_generator(i)
        model = next(train_gen)
        for k, v in model.networks.items():
            if k in all_networks.keys() and k not in shared_networks:
                shared_networks.append(k)
            all_networks[k] = v.module
        trainers.append(train_gen)
    print("Networks being shared by trainers: ", shared_networks)

    # Now, simply "iterate" through the trainers to accomplish training.
    while True:
        for trainer in trainers:
            next(trainer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.', default='../options/train_exd_imgset_chained_structured_trans_invariance.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    Loader, Dumper = OrderedYaml()
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
        main(opt, args.launcher)
