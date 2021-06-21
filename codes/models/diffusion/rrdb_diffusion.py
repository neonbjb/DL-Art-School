import torch
import torch.nn as nn
import torch.nn.functional as F

from models.arch_util import ConvGnLelu, default_init_weights, make_layer
from models.diffusion.nn import timestep_embedding
from trainer.networks import register_model
from utils.util import checkpoint


# Conditionally uses torch's checkpoint functionality if it is enabled in the opt file.


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels=64, growth_channels=32, embedding=False, init_weight=.1):
        super(ResidualDenseBlock, self).__init__()
        self.embedding = embedding
        if embedding:
            self.first_conv = ConvGnLelu(mid_channels, mid_channels, activation=True, norm=False, bias=True)
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    mid_channels*4,
                    mid_channels,
                ),
            )
        for i in range(5):
            out_channels = mid_channels if i == 4 else growth_channels
            self.add_module(
                f'conv{i + 1}',
                nn.Conv2d(mid_channels + i * growth_channels, out_channels, 3,
                          1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        for i in range(5):
            default_init_weights(getattr(self, f'conv{i + 1}'), init_weight)
        default_init_weights(self.conv5, 0)

        self.normalize = nn.GroupNorm(num_groups=8, num_channels=mid_channels)

    def forward(self, x, emb):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.embedding:
            x0 = self.first_conv(x)
            emb_out = self.emb_layers(emb).type(x0.dtype)
            while len(emb_out.shape) < len(x0.shape):
                emb_out = emb_out[..., None]
            x0 = x0 + emb_out
        else:
            x0 = x
        x1 = self.lrelu(self.conv1(x0))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return self.normalize(x5 * .2 + x)


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels, growth_channels=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(mid_channels, growth_channels, embedding=True)
        self.rdb2 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(mid_channels, growth_channels)
        self.normalize = nn.GroupNorm(num_groups=8, num_channels=mid_channels)
        self.residual_mult = nn.Parameter(torch.FloatTensor([.1]))

    def forward(self, x, emb):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = self.rdb1(x, emb)
        out = self.rdb2(out, emb)
        out = self.rdb3(out, emb)

        return self.normalize(out * self.residual_mult + x)


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Currently, it supports x4 upsampling scale factor.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Defaults: 23
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_blocks=23,
                 growth_channels=32,
                 body_block=RRDB,
                 ):
        super(RRDBNet, self).__init__()
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        # The diffusion RRDB starts with a full resolution image and downsamples into a .25 working space
        self.input_block = ConvGnLelu(in_channels, mid_channels, kernel_size=7, stride=1, activation=True, norm=False, bias=True)
        self.down1 = ConvGnLelu(mid_channels, mid_channels, kernel_size=3, stride=2, activation=True, norm=False, bias=True)
        self.down2 = ConvGnLelu(mid_channels, mid_channels, kernel_size=3, stride=2, activation=True, norm=False, bias=True)

        # Guided diffusion uses a time embedding.
        time_embed_dim = mid_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(mid_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.body = make_layer(
            body_block,
            num_blocks,
            mid_channels=mid_channels,
            growth_channels=growth_channels)

        self.conv_body = nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(self.mid_channels, self.mid_channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(self.mid_channels*2, self.mid_channels, 3, 1, 1)
        self.conv_up3 = None
        self.conv_hr = nn.Conv2d(self.mid_channels*2, self.mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(self.mid_channels, out_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.normalize = nn.GroupNorm(num_groups=8, num_channels=self.mid_channels)

        for m in [
            self.conv_body, self.conv_up1,
            self.conv_up2, self.conv_hr
        ]:
            if m is not None:
                default_init_weights(m, 1.0)
        default_init_weights(self.conv_last, 0)

    def forward(self, x, timesteps, low_res, correction_factors=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.mid_channels))

        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = torch.cat([x, upsampled], dim=1)

        if correction_factors is not None:
            correction_factors = correction_factors.view(x.shape[0], -1, 1, 1).repeat(1, 1, new_height, new_width)
        else:
            correction_factors = torch.zeros((b, self.num_corruptions, new_height, new_width), dtype=torch.float, device=x.device)
        x = torch.cat([x, correction_factors], dim=1)

        d1 = self.input_block(x)
        d2 = self.down1(d1)
        feat = self.down2(d2)
        for bl in self.body:
            feat = checkpoint(bl, feat, emb)
        feat = feat[:, :self.mid_channels]
        feat = self.conv_body(feat)

        # upsample
        out = torch.cat([self.lrelu(
            self.normalize(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))),
            d2], dim=1)
        out = torch.cat([self.lrelu(
            self.normalize(self.conv_up2(F.interpolate(out, scale_factor=2, mode='nearest')))),
            d1], dim=1)
        out = self.conv_last(self.normalize(self.lrelu(self.conv_hr(out))))

        return out


@register_model
def register_rrdb_diffusion(opt_net, opt):
    return RRDBNet(**opt_net['args'])


if __name__ == '__main__':
    model = RRDBNet(6,6)
    x = torch.randn(1,3,128,128)
    l = torch.randn(1,3,32,32)
    t = torch.LongTensor([555])
    y = model(x, t, l)
    print(y.shape, y.mean(), y.std(), y.min(), y.max())
