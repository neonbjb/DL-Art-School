import argparse
import options.options as option
from models.networks import define_G
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YMAL file.', default='options/test/test_ESRGAN_adrianna_full.yml')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    netG = define_G(opt)

    print("Tracing generator network..")
    dummyInput = torch.rand(1, 3, 8, 8)
    traced_netG = torch.jit.trace(netG, dummyInput)
    traced_netG.save('../results/traced_generator.zip')
    print(traced_netG)