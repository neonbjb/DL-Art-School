import torch
from torch.cuda.amp import autocast
from models.flownet2.networks.resample2d_package.resample2d import Resample2d
from models.steps.injectors import Injector


def create_stereoscopic_injector(opt, env):
    type = opt['type']
    if type == 'stereoscopic_resample':
        return ResampleInjector(opt, env)
    return None


class ResampleInjector(Injector):
    def __init__(self, opt, env):
        super(ResampleInjector, self).__init__(opt, env)
        self.resample = Resample2d()
        self.flow = opt['flowfield']

    def forward(self, state):
        with autocast(enabled=False):
            return {self.output: self.resample(state[self.input], state[self.flow])}