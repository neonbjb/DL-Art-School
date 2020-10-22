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
import train
import utils.options as option

def main(master_opt, launcher):
    trainers = []
    all_networks = {}
    shared_networks = []
    for i, sub_opt in enumerate(master_opt['trainer_options']):
        sub_opt_parsed = option.parse(sub_opt, is_train=True)
        # This creates trainers() as a list of generators.
        train_gen = train.yielding_main(sub_opt_parsed, launcher, i, all_networks)
        model = next(train_gen)
        for k, v in model.networks.items():
            if k in all_networks.keys() and k not in shared_networks:
                shared_networks.append(k)
            all_networks[k] = v
        trainers.append(train_gen)
    print("Networks being shared by trainers: ", shared_networks)

    # Now, simply "iterate" through the trainers to accomplish training.
    while True:
        for trainer in trainers:
            next(trainer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-opt', type=str, help='Path to option YAML file.', default='../options/train_exd_imgset_chained_structured_trans_invariance.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    args = parser.parse_args()
    opt = {
        'trainer_options': ['../options/teco.yml', '../options/exd.yml']
    }
    main(opt, args.launcher)