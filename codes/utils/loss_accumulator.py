import torch

# Utility class that stores detached, named losses in a rotating buffer for smooth metric outputting.
class LossAccumulator:
    def __init__(self, buffer_sz=50):
        self.buffer_sz = buffer_sz
        self.buffers = {}

    def add_loss(self, name, tensor):
        if name not in self.buffers.keys():
            self.buffers[name] = (0, torch.zeros(self.buffer_sz))
        i, buf = self.buffers[name]
        # Can take tensors or just plain python numbers.
        if isinstance(tensor, torch.Tensor):
            buf[i] = tensor.detach().cpu()
        else:
            buf[i] = tensor
        self.buffers[name] = ((i+1) % self.buffer_sz, buf)

    def as_dict(self):
        result = {}
        for k, v in self.buffers.items():
            result["loss_" + k] = torch.mean(v[1])
        return result