import torch

# Utility class that stores detached, named losses in a rotating buffer for smooth metric outputting.
from torch import distributed


class LossAccumulator:
    def __init__(self, buffer_sz=50):
        self.buffer_sz = buffer_sz
        self.buffers = {}
        self.counters = {}

    def add_loss(self, name, tensor):
        if name not in self.buffers.keys():
            if "_histogram" in name:
                tensor = torch.flatten(tensor.detach().cpu())
                self.buffers[name] = (0, torch.zeros((self.buffer_sz, tensor.shape[0])), False)
            else:
                self.buffers[name] = (0, torch.zeros(self.buffer_sz), False)
        i, buf, filled = self.buffers[name]
        # Can take tensors or just plain python numbers.
        if '_histogram' in name:
            buf[i] = torch.flatten(tensor.detach().cpu())
        elif isinstance(tensor, torch.Tensor):
            buf[i] = tensor.detach().cpu()
        else:
            buf[i] = tensor
        filled = i+1 >= self.buffer_sz or filled
        self.buffers[name] = ((i+1) % self.buffer_sz, buf, filled)

    def increment_metric(self, name):
        if name not in self.counters.keys():
            self.counters[name] = 1
        else:
            self.counters[name] += 1

    def as_dict(self):
        result = {}
        for k, v in self.buffers.items():
            i, buf, filled = v
            if '_histogram' in k:
                result["loss_" + k] = torch.flatten(buf)
            if filled:
                result["loss_" + k] = torch.mean(buf)
            else:
                result["loss_" + k] = torch.mean(buf[:i])
        for k, v in self.counters.items():
            result[k] = v
        return result


# Stores losses in an infinitely-sized list.
class InfStorageLossAccumulator:
    def __init__(self):
        self.buffers = {}

    def add_loss(self, name, tensor):
        if name not in self.buffers.keys():
            if "_histogram" in name:
                tensor = torch.flatten(tensor.detach().cpu())
                self.buffers[name] = []
            else:
                self.buffers[name] = []
        buf = self.buffers[name]
        # Can take tensors or just plain python numbers.
        if '_histogram' in name:
            buf.append(torch.flatten(tensor.detach().cpu()))
        elif isinstance(tensor, torch.Tensor):
            buf.append(tensor.detach().cpu())
        else:
            buf.append(tensor)

    def increment_metric(self, name):
        pass

    def as_dict(self):
        result = {}
        for k, buf in self.buffers.items():
            if '_histogram' in k:
                result["loss_" + k] = torch.flatten(buf)
            else:
                result["loss_" + k] = torch.mean(torch.stack(buf))
        return result