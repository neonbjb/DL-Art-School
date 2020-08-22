import torch.nn
from models.archs.SPSR_arch import ImageGradientNoPadding

# Injectors are a way to sythesize data within a step that can then be used (and reused) by loss functions.
def create_injector(opt_inject, env):
    type = opt_inject['type']
    if type == 'img_grad':
        return ImageGradientInjector(opt_inject, env)
    else:
        raise NotImplementedError


class Injector(torch.nn.Module):
    def __init__(self, opt, env):
        super(self, Injector).__init__()
        self.opt = opt
        self.env = env
        self.input = opt['in']
        self.output = opt['out']

    # This should return a dict of new state variables.
    def forward(self, state):
        raise NotImplementedError


class ImageGradientInjector(Injector):
    def __init__(self, opt, env):
        super(self, ImageGradientInjector).__init__(opt, env)
        self.img_grad_fn = ImageGradientNoPadding()

    def forward(self, state):
        return {self.opt['out']: self.img_grad_fn(state[self.opt['in']])}