import argparse
import options.options as option
from models.networks import define_G
import torch
import torchvision
import torch.nn.functional as F


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../options/use_video_upsample.yml')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    netG = define_G(opt)
    dummyInput = torch.rand(1,3,8,8)

    torchscript = False
    if torchscript:
        print("Tracing generator network..")
        traced_netG = torch.jit.trace(netG, dummyInput)
        traced_netG.save('../results/ts_generator.zip')
        print(traced_netG)
    else:
        print("Performing onnx trace")
        input_names = ["lr_input"]
        output_names = ["hr_image"]
        dynamic_axes = {'lr_input': {0: 'batch', 1: 'filters', 2: 'h', 3: 'w'}, 'hr_image': {0: 'batch', 1: 'filters', 2: 'h', 3: 'w'}}

        torch.onnx.export(netG, dummyInput, "../results/gen.onnx", verbose=True, input_names=input_names,
                          output_names=output_names, dynamic_axes=dynamic_axes, opset_version=11)