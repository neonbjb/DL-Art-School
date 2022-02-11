import functools
import os
from multiprocessing.pool import ThreadPool

import torch

from train import Trainer
from utils import options as option

def launch_trainer(opt, opt_path=''):
    rank = opt['gpu_ids'][0]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    print('export CUDA_VISIBLE_DEVICES=' + str(rank))
    trainer = Trainer()
    opt['dist'] = False
    trainer.rank = -1
    torch.cuda.set_device(rank)
    trainer.init(opt_path, opt, 'none')
    trainer.do_training()

if __name__ == '__main__':
    """
    Ad-hoc script (hard coded; no command-line parameters) that spawns multiple separate trainers from a single options
    file, with a hard-coded set of modifications.
    """
    base_opt = '../experiments/train_diffusion_tts6.yml'
    modifications = {
        'baseline': {},
        'only_conv': {'networks': {'generator': {'kwargs': {'cond_transformer_depth': 4, 'mid_transformer_depth': 1}}}},
        'intermediary_attention': {'networks': {'generator': {'kwargs': {'attention_resolutions': [32,64], 'num_res_blocks': [2, 2, 2, 2, 2, 2, 2]}}}},
        'more_resblocks': {'networks': {'generator': {'kwargs': {'num_res_blocks': [3, 3, 3, 3, 3, 3, 2]}}}},
        'less_resblocks': {'networks': {'generator': {'kwargs': {'num_res_blocks': [1, 1, 1, 1, 1, 1, 1]}}}},
        'wider': {'networks': {'generator': {'kwargs': {'channel_mult': [1,2,4,6,8,8,8]}}}},
        'inject_every_layer': {'networks': {'generator': {'kwargs': {'token_conditioning_resolutions': [1,2,4,8,16,32,64]}}}},
        'cosine_diffusion': {'steps': {'generator': {'injectors': {'diffusion': {'beta_schedule': {'schedule_name': 'cosine'}}}}}},
    }
    opt = option.parse(base_opt, is_train=True)
    all_opts = []
    for i, (mod, mod_dict) in enumerate(modifications.items()):
        nd = opt.copy()
        nd.update(mod_dict)
        opt['gpu_ids'] = [i]
        nd['name'] = f'{nd["name"]}_{mod}'
        nd['wandb_run_name'] = mod
        base_path = nd['path']['log']
        for k, p in nd['path'].items():
            if isinstance(p, str) and base_path in p:
                nd['path'][k] = p.replace(base_path, f'{base_path}\\{mod}')
        all_opts.append(nd)

    with ThreadPool(len(modifications)) as pool:
        list(pool.imap(functools.partial(launch_trainer, opt_path=base_opt), all_opts))
