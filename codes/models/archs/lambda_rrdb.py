import torch
from torch import nn
from lambda_networks import LambdaLayer
from torch.nn import GroupNorm

from models.archs.RRDBNet_arch import ResidualDenseBlock
from models.archs.arch_util import ConvGnLelu


class LambdaRRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels, growth_channels=32, reduce_to=None):
        super(LambdaRRDB, self).__init__()
        if reduce_to is None:
            reduce_to = mid_channels
        self.lam1 = LambdaLayer(dim=mid_channels, dim_out=mid_channels, r=23, dim_k=16, heads=4, dim_u=4)
        self.gn1 = GroupNorm(num_groups=8, num_channels=mid_channels)
        self.lam2 = LambdaLayer(dim=mid_channels, dim_out=mid_channels, r=23, dim_k=16, heads=4, dim_u=4)
        self.gn2 = GroupNorm(num_groups=8, num_channels=mid_channels)
        self.lam3 = LambdaLayer(dim=mid_channels, dim_out=reduce_to, r=23, dim_k=16, heads=4, dim_u=4)
        self.gn3 = GroupNorm(num_groups=8, num_channels=mid_channels)
        self.conv = ConvGnLelu(reduce_to, reduce_to, kernel_size=1, bias=True, norm=False, activation=False, weight_init_factor=.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = self.lam1(x)
        out = self.gn1(out)
        out = self.lam2(out)
        out = self.gn1(out)
        out = self.lam3(out)
        out = self.gn3(out)
        return self.conv(out) * .2 + x