# Converts from Tensorflow Stylegan2 weights to weights used by this model.
# Original source: https://raw.githubusercontent.com/rosinality/stylegan2-pytorch/master/convert_weight.py
# Adapted to lucidrains' Stylegan implementation.
#
# Also doesn't require you to install Tensorflow 1.15 or clone the nVidia repo.

# THIS DOES NOT CURRENTLY WORK.
# It does transfer all weights from the stylegan model to the lucidrains one, but does not produce correct results.
# The rosinality script this was stolen from has some "odd" intracacies that may be at cause for this: for example
# weight "flipping" in the conv layers which I do not understand. It may also be because I botched some of the mods
# required to make the lucidrains implementation conformant. I'll (maybe) get back to this some day.

import argparse
import os
import sys
import pickle
import math

import torch
import numpy as np
from torchvision import utils


# Converts from the TF state_dict input provided into the vars originally expected from the rosinality converter.
from models.stylegan.stylegan2_lucidrains import StyleGan2GeneratorWithLatent


def get_vars(vars, source_name):
    net_name = source_name.split('/')[0]
    vars_as_tuple_list = vars[net_name]['variables']
    result_vars = {}
    for t in vars_as_tuple_list:
        result_vars[t[0]] = t[1]
    return result_vars, source_name.replace(net_name + "/", "")

def get_vars_direct(vars, source_name):
    v, n = get_vars(vars, source_name)
    return v[n]


def convert_modconv(vars, source_name, target_name, flip=False, numeral=1):
    vars, source_name = get_vars(vars, source_name)
    weight = vars[source_name + "/weight"]
    mod_weight = vars[source_name + "/mod_weight"]
    mod_bias = vars[source_name + "/mod_bias"]
    noise = vars[source_name + "/noise_strength"]
    bias = vars[source_name + "/bias"]

    dic = {
        f"conv{numeral}.weight": weight.transpose((3, 2, 0, 1)),
        f"to_style{numeral}.weight": mod_weight.transpose((1, 0)),
        f"to_style{numeral}.bias": mod_bias + 1,
        f"noise{numeral}_scale": np.array([noise]),
        f"activation{numeral}.bias": bias,
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + "." + k] = torch.from_numpy(v)

    if flip:
        dic_torch[target_name + f".conv{numeral}.weight"] = torch.flip(
            dic_torch[target_name + f".conv{numeral}.weight"], [2, 3]
        )

    return dic_torch


def convert_conv(vars, source_name, target_name, bias=True, start=0):
    vars, source_name = get_vars(vars, source_name)
    weight = vars[source_name + "/weight"]

    dic = {"weight": weight.transpose((3, 2, 0, 1))}

    if bias:
        dic["bias"] = vars[source_name + "/bias"]

    dic_torch = {}

    dic_torch[target_name + f".{start}.weight"] = torch.from_numpy(dic["weight"])

    if bias:
        dic_torch[target_name + f".{start + 1}.bias"] = torch.from_numpy(dic["bias"])

    return dic_torch


def convert_torgb(vars, source_name, target_name):
    vars, source_name = get_vars(vars, source_name)
    weight = vars[source_name + "/weight"]
    mod_weight = vars[source_name + "/mod_weight"]
    mod_bias = vars[source_name + "/mod_bias"]
    bias = vars[source_name + "/bias"]

    dic = {
        "conv.weight": weight.transpose((3, 2, 0, 1)),
        "to_style.weight": mod_weight.transpose((1, 0)),
        "to_style.bias": mod_bias + 1,
        # "bias": bias.reshape((1, 3, 1, 1)), TODO: where is this?
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + "." + k] = torch.from_numpy(v)

    return dic_torch


def convert_dense(vars, source_name, target_name):
    vars, source_name = get_vars(vars, source_name)
    weight = vars[source_name + "/weight"]
    bias = vars[source_name + "/bias"]

    dic = {"weight": weight.transpose((1, 0)), "bias": bias}

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + "." + k] = torch.from_numpy(v)

    return dic_torch


def update(state_dict, new, strict=True):

    for k, v in new.items():
        if strict:
            if k not in state_dict:
                raise KeyError(k + " is not found")

            if v.shape != state_dict[k].shape:
                raise ValueError(f"Shape mismatch: {v.shape} vs {state_dict[k].shape}")

        state_dict[k] = v


def discriminator_fill_statedict(statedict, vars, size):
    log_size = int(math.log(size, 2))

    update(statedict, convert_conv(vars, f"{size}x{size}/FromRGB", "convs.0"))

    conv_i = 1

    for i in range(log_size - 2, 0, -1):
        reso = 4 * 2 ** i
        update(
            statedict,
            convert_conv(vars, f"{reso}x{reso}/Conv0", f"convs.{conv_i}.conv1"),
        )
        update(
            statedict,
            convert_conv(
                vars, f"{reso}x{reso}/Conv1_down", f"convs.{conv_i}.conv2", start=1
            ),
        )
        update(
            statedict,
            convert_conv(
                vars, f"{reso}x{reso}/Skip", f"convs.{conv_i}.skip", start=1, bias=False
            ),
        )
        conv_i += 1

    update(statedict, convert_conv(vars, f"4x4/Conv", "final_conv"))
    update(statedict, convert_dense(vars, f"4x4/Dense0", "final_linear.0"))
    update(statedict, convert_dense(vars, f"Output", "final_linear.1"))

    return statedict


def fill_statedict(state_dict, vars, size):
    log_size = int(math.log(size, 2))

    for i in range(8):
        update(state_dict, convert_dense(vars, f"G_mapping/Dense{i}", f"vectorizer.net.{i}"))

    update(
        state_dict,
        {
            "gen.initial_block": torch.from_numpy(
                get_vars_direct(vars, "G_synthesis/4x4/Const/const")
            )
        },
    )

    for i in range(log_size - 1):
        reso = 4 * 2 ** i
        update(
            state_dict,
            convert_torgb(vars, f"G_synthesis/{reso}x{reso}/ToRGB", f"gen.blocks.{i}.to_rgb"),
        )

    update(state_dict, convert_modconv(vars, "G_synthesis/4x4/Conv", "gen.blocks.0", numeral=1))

    for i in range(1, log_size - 1):
        reso = 4 * 2 ** i
        update(
            state_dict,
            convert_modconv(
                vars,
                f"G_synthesis/{reso}x{reso}/Conv0_up",
                f"gen.blocks.{i}",
                #flip=True,  # TODO: why??
                numeral=1
            ),
        )
        update(
            state_dict,
            convert_modconv(
                vars, f"G_synthesis/{reso}x{reso}/Conv1", f"gen.blocks.{i}", numeral=2
            ),
        )

    '''
    TODO: consider porting this, though I dont think it is necessary.
    for i in range(0, (log_size - 2) * 2 + 1):
        update(
            state_dict,
            {
                f"noises.noise_{i}": torch.from_numpy(
                    get_vars_direct(vars, f"G_synthesis/noise{i}")
                )
            },
        )
    '''

    return state_dict


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Tensorflow to pytorch model checkpoint converter"
    )
    parser.add_argument(
        "--gen", action="store_true", help="convert the generator weights"
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor. config-f = 2, else = 1",
    )
    parser.add_argument("path", metavar="PATH", help="path to the tensorflow weights")

    args = parser.parse_args()
    sys.path.append('scripts\\stylegan2')

    import dnnlib
    from dnnlib.tflib.network import generator, gen_ema

    with open(args.path, "rb") as f:
        pickle.load(f)

    # Weight names are ordered by size. The last name will be something like '1024x1024/<blah>'. We just need to grab that first number.
    size = int(generator['G_synthesis']['variables'][-1][0].split('x')[0])

    g = StyleGan2GeneratorWithLatent(image_size=size, latent_dim=512, style_depth=8)
    state_dict = g.state_dict()
    state_dict = fill_statedict(state_dict, gen_ema, size)

    g.load_state_dict(state_dict, strict=True)

    latent_avg = torch.from_numpy(get_vars_direct(gen_ema, "G/dlatent_avg"))

    ckpt = {"g_ema": state_dict, "latent_avg": latent_avg}

    if args.gen:
        g_train = Generator(size, 512, 8, channel_multiplier=args.channel_multiplier)
        g_train_state = g_train.state_dict()
        g_train_state = fill_statedict(g_train_state, generator, size)
        ckpt["g"] = g_train_state

    name = os.path.splitext(os.path.basename(args.path))[0]
    torch.save(ckpt, name + ".pt")

    batch_size = {256: 16, 512: 9, 1024: 4}
    n_sample = batch_size.get(size, 25)

    g = g.to(device)

    z = np.random.RandomState(5).randn(n_sample, 512).astype("float32")

    with torch.no_grad():
        img_pt, _ = g(8)

    utils.save_image(
        img_pt, name + ".png", nrow=n_sample, normalize=True, range=(-1, 1)
    )
