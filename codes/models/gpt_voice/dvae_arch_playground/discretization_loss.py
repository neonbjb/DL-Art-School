import random
from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F


# Fits a soft-discretized input to a normal-PDF across the specified dimension.
# In other words, attempts to force the discretization function to have a mean equal utilization across all discrete
#  values with the specified expected variance.
class DiscretizationLoss(nn.Module):
    def __init__(self, discrete_bins, dim, expected_variance, store_past=0):
        super().__init__()
        self.discrete_bins = discrete_bins
        self.dim = dim
        self.dist = torch.distributions.Normal(0, scale=expected_variance)
        if store_past > 0:
            self.record_past = True
            self.register_buffer("accumulator_index", torch.zeros(1, dtype=torch.long, device='cpu'))
            self.register_buffer("accumulator_filled", torch.zeros(1, dtype=torch.long, device='cpu'))
            self.register_buffer("accumulator", torch.zeros(store_past, discrete_bins))
        else:
            self.record_past = False

    def forward(self, x):
        other_dims = set(range(len(x.shape)))-set([self.dim])
        averaged = x.sum(dim=tuple(other_dims)) / x.sum()
        averaged = averaged - averaged.mean()

        if self.record_past:
            acc_count = self.accumulator.shape[0]
            avg = averaged.detach().clone()
            if self.accumulator_filled > 0:
                averaged = torch.mean(self.accumulator, dim=0) * (acc_count-1) / acc_count + \
                           averaged / acc_count

            # Also push averaged into the accumulator.
            self.accumulator[self.accumulator_index] = avg
            self.accumulator_index += 1
            if self.accumulator_index >= acc_count:
                self.accumulator_index *= 0
                if self.accumulator_filled <= 0:
                    self.accumulator_filled += 1

        return torch.sum(-self.dist.log_prob(averaged))


if __name__ == '__main__':
    d = DiscretizationLoss(1024, 1, 1e-6, store_past=20)
    for _ in range(500):
        v = torch.randn(16, 1024, 500)
        #for k in range(5):
        #    v[:, random.randint(0,8192), :] += random.random()*100
        v = F.softmax(v, 1)
        print(d(v))