# Converts from Tensorflow Stylegan2 weights to weights used by this model.
# Original source: https://raw.githubusercontent.com/rosinality/stylegan2-pytorch/master/convert_weight.py
#
# Also doesn't require you to install Tensorflow 1.15 or clone the nVidia repo.

import argparse
import os
import sys
import pickle
import math

import torch
import numpy as np
from torchvision import utils

from models.image_generation.stylegan.stylegan2_rosinality import Generator, Discriminator


# Converts from the TF state_dict input provided into the vars originally expected from the rosinality converter.
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


def convert_modconv(vars, source_name, target_name, flip=False):
    vars, source_name = get_vars(vars, source_name)
    weight = vars[source_name + "/weight"]
    mod_weight = vars[source_name + "/mod_weight"]
    mod_bias = vars[source_name + "/mod_bias"]
    noise = vars[source_name + "/noise_strength"]
    bias = vars[source_name + "/bias"]

    dic = {
        "conv.weight": np.expand_dims(weight.transpose((3, 2, 0, 1)), 0),
        "conv.modulation.weight": mod_weight.transpose((1, 0)),
        "conv.modulation.bias": mod_bias + 1,
        "noise.weight": np.array([noise]),
        "activate.bias": bias,
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + "." + k] = torch.from_numpy(v)

    if flip:
        dic_torch[target_name + ".conv.weight"] = torch.flip(
            dic_torch[target_name + ".conv.weight"], [3, 4]
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
        "conv.weight": np.expand_dims(weight.transpose((3, 2, 0, 1)), 0),
        "conv.modulation.weight": mod_weight.transpose((1, 0)),
        "conv.modulation.bias": mod_bias + 1,
        "bias": bias.reshape((1, 3, 1, 1)),
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


def update(state_dict, new):
    for k, v in new.items():
        state_dict[k] = v


def discriminator_fill_statedict(statedict, vars, size):
    log_size = int(math.log(size, 2))

    update(statedict, convert_conv(vars, f"D/{size}x{size}/FromRGB", "convs.0"))

    conv_i = 1

    for i in range(log_size - 2, 0, -1):
        reso = 4 * 2 ** i
        update(
            statedict,
            convert_conv(vars, f"D/{reso}x{reso}/Conv0", f"convs.{conv_i}.conv1"),
        )
        update(
            statedict,
            convert_conv(
                vars, f"D/{reso}x{reso}/Conv1_down", f"convs.{conv_i}.conv2", start=1
            ),
        )
        update(
            statedict,
            convert_conv(
                vars, f"D/{reso}x{reso}/Skip", f"convs.{conv_i}.skip", start=1, bias=False
            ),
        )
        conv_i += 1

    update(statedict, convert_conv(vars, f"D/4x4/Conv", "final_conv"))
    update(statedict, convert_dense(vars, f"D/4x4/Dense0", "final_linear.0"))
    update(statedict, convert_dense(vars, f"D/Output", "final_linear.1"))

    return statedict


def fill_statedict(state_dict, vars, size):
    log_size = int(math.log(size, 2))

    for i in range(8):
        update(state_dict, convert_dense(vars, f"G_mapping/Dense{i}", f"style.{i + 1}"))

    update(
        state_dict,
        {
            "input.input": torch.from_numpy(
                get_vars_direct(vars, "G_synthesis/4x4/Const/const")
            )
        },
    )

    update(state_dict, convert_torgb(vars, "G_synthesis/4x4/ToRGB", "to_rgb1"))

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        update(
            state_dict,
            convert_torgb(vars, f"G_synthesis/{reso}x{reso}/ToRGB", f"to_rgbs.{i}"),
        )

    update(state_dict, convert_modconv(vars, "G_synthesis/4x4/Conv", "conv1"))

    conv_i = 0

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        update(
            state_dict,
            convert_modconv(
                vars,
                f"G_synthesis/{reso}x{reso}/Conv0_up",
                f"convs.{conv_i}",
                flip=True,
            ),
        )
        update(
            state_dict,
            convert_modconv(
                vars, f"G_synthesis/{reso}x{reso}/Conv1", f"convs.{conv_i + 1}"
            ),
        )
        conv_i += 2

    for i in range(0, (log_size - 2) * 2 + 1):
        update(
            state_dict,
            {
                f"noises.noise_{i}": torch.from_numpy(
                    get_vars_direct(vars, f"G_synthesis/noise{i}")
                )
            },
        )

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
        "--disc", action="store_true", help="convert the discriminator weights"
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

    from dnnlib.tflib.network import generator, discriminator, gen_ema

    with open(args.path, "rb") as f:
        pickle.load(f)

    # Weight names are ordered by size. The last name will be something like '1024x1024/<blah>'. We just need to grab that first number.
    size = int(generator['G_synthesis']['variables'][-1][0].split('x')[0])

    g = Generator(size, 512, 8, channel_multiplier=args.channel_multiplier)
    state_dict = g.state_dict()
    state_dict = fill_statedict(state_dict, gen_ema, size)
    g.load_state_dict(state_dict, strict=True)

    d = Discriminator(size, args.channel_multiplier)
    dstate_dict = d.state_dict()
    dstate_dict = discriminator_fill_statedict(dstate_dict, discriminator, size)
    d.load_state_dict(dstate_dict, strict=True)


    latent_avg = torch.from_numpy(get_vars_direct(gen_ema, "G/dlatent_avg"))

    ckpt = {"g_ema": state_dict, "latent_avg": latent_avg}

    if args.gen:
        g_train = Generator(size, 512, 8, channel_multiplier=args.channel_multiplier)
        g_train_state = g_train.state_dict()
        g_train_state = fill_statedict(g_train_state, generator, size)
        ckpt["g"] = g_train_state

    if args.disc:
        disc = Discriminator(size, channel_multiplier=args.channel_multiplier)
        d_state = disc.state_dict()
        d_state = discriminator_fill_statedict(d_state, discriminator.vars, size)
        ckpt["d"] = d_state

    name = os.path.splitext(os.path.basename(args.path))[0]
    torch.save(state_dict, f"{name}_gen.pth")
    torch.save(dstate_dict, f"{name}_disc.pth")

    batch_size = {256: 16, 512: 9, 1024: 4}
    n_sample = batch_size.get(size, 25)

    g = g.to(device)
    d = d.to(device)

    z = np.random.RandomState(1).randn(n_sample, 512).astype("float32")

    with torch.no_grad():
        img_pt, _ = g(
            [torch.from_numpy(z).to(device)],
            truncation=0.5,
            truncation_latent=latent_avg.to(device),
            randomize_noise=False,
        )
        disc = d(img_pt)
        print(disc)

    utils.save_image(
        img_pt, name + ".png", nrow=n_sample, normalize=True, range=(-1, 1)
    )
