import torch
from torch.cuda.amp import autocast
from models.flownet2.networks import Resample2d
from models.flownet2 import flow2img
from trainer.inject import Injector


def create_stereoscopic_injector(opt, env):
    type = opt['type']
    if type == 'stereoscopic_resample':
        return ResampleInjector(opt, env)
    elif type == 'stereoscopic_flow2image':
        return Flow2Image(opt, env)
    return None


class ResampleInjector(Injector):
    def __init__(self, opt, env):
        super(ResampleInjector, self).__init__(opt, env)
        self.resample = Resample2d()
        self.flow = opt['flowfield']

    def forward(self, state):
        with autocast(enabled=False):
            return {self.output: self.resample(state[self.input], state[self.flow])}


# Converts a flowfield to an image representation for viewing purposes.
# Uses flownet's implementation to do so. Which really sucks. TODO: just do my own implementation in the future.
# Note: this is not differentiable and is only usable for debugging purposes.
class Flow2Image(Injector):
    def __init__(self, opt, env):
        super(Flow2Image, self).__init__(opt, env)

    def forward(self, state):
        with torch.no_grad():
            flo = state[self.input].cpu()
            bs, c, h, w = flo.shape
            flo = flo.permute(0, 2, 3, 1)  # flow2img works in numpy space for some reason..
            imgs = torch.empty_like(flo)
            flo = flo.numpy()
            for b in range(bs):
                img = flow2img(flo[b])  # Note that this returns the image in an integer format.
                img = torch.tensor(img, dtype=torch.float) / 255
                imgs[b] = img
            imgs = imgs.permute(0, 3, 1, 2)
            return {self.output: imgs}
