import torch

# Utility class that stores detached, named losses in a rotating buffer for smooth metric outputting.
class LossAccumulator:
    def __init__(self, buffer_sz=10):
        self.buffer_sz = buffer_sz
        self.buffers = {}

    def add_loss(self, name, tensor):
        if name not in self.buffers.keys():
            self.buffers[name] = (0, torch.zeros(self.buffer_sz))
        i, buf = self.buffers[name]
        buf[i] = tensor.detach().cpu()
        self.buffers[name] = ((i+1) % self.buffer_sz, buf)

    def as_dict(self):
        result = {}
        for k, v in self.buffers:
            result["loss_" + k] = torch.mean(v)
        return result