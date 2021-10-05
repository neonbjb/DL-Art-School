import random
from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F


# Fits a soft-discretized input to a normal-PDF across the specified dimension.
# In other words, attempts to force the discretization function to have a mean equal utilization across all discrete
#  values with the specified expected variance.
class DiscretizationLoss(nn.Module):
    def __init__(self, dim, expected_variance):
        super().__init__()
        self.dim = dim
        self.dist = torch.distributions.Normal(0, scale=expected_variance)

    def forward(self, x):
        other_dims = set(range(len(x.shape)))-set([self.dim])
        averaged = x.sum(dim=tuple(other_dims)) / x.sum()
        averaged = averaged - averaged.mean()
        return torch.sum(-self.dist.log_prob(averaged))


if __name__ == '__main__':
    d = DiscretizationLoss(1, 1e-6)
    v = torch.randn(16, 8192, 500)
    #for k in range(5):
    #    v[:, random.randint(0,8192), :] += random.random()*100
    v = F.softmax(v, 1)
    print(d(v))