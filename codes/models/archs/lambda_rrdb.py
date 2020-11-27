import torch
from torch import nn
from lambda_networks import LambdaLayer
from torch.nn import GroupNorm

from models.archs.RRDBNet_arch import ResidualDenseBlock


class LambdaRRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels, growth_channels=32, reduce_to=None):
        super(LambdaRRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(mid_channels, growth_channels, init_weight=1)
        self.rdb2 = ResidualDenseBlock(mid_channels, growth_channels, init_weight=1)
        if reduce_to is None:
            reduce_to = mid_channels
        self.lam = LambdaLayer(dim=mid_channels, dim_out=reduce_to, r=23, dim_k=16, heads=4, dim_u=4)
        self.gn = GroupNorm(num_groups=8, num_channels=mid_channels)
        self.scale = nn.Parameter(torch.full((1,), 1/256))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.lam(out)
        out = self.gn(out)
        return out * self.scale + x