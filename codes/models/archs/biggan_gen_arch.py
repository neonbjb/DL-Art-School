# Source: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/BigGANdeep.py
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import models.archs.biggan_layers as layers

# BigGAN-deep: uses a different resblock and pattern

# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.

# Channel ratio is the ratio of
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d, which_bn=layers.bn, activation=None,
                 upsample=None, channel_ratio=4):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.in_channels // channel_ratio
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels,
                                     kernel_size=1, padding=0)
        self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv4 = self.which_conv(self.hidden_channels, self.out_channels,
                                     kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(self.in_channels)
        self.bn2 = self.which_bn(self.hidden_channels)
        self.bn3 = self.which_bn(self.hidden_channels)
        self.bn4 = self.which_bn(self.hidden_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x):
        # Project down to channel ratio
        h = self.conv1(self.activation(self.bn1(x)))
        # Apply next BN-ReLU
        h = self.activation(self.bn2(h))
        # Drop channels in x if necessary
        if self.in_channels != self.out_channels:
            x = x[:, :self.out_channels]
            # Upsample both h and x at this point
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        # 3x3 convs
        h = self.conv2(h)
        h = self.conv3(self.activation(self.bn3(h)))
        # Final 1x1 conv
        h = self.conv4(self.activation(self.bn4(h)))
        return h + x


def G_arch(ch=64, attention='64', base_width=64):
    arch = {}
    arch[128] = {'in_channels': [ch * item for item in [2, 2, 1, 1]],
                 'out_channels': [ch * item for item in [2, 1, 1, 1]],
                 'upsample': [False, True, False, False],
                 'resolution': [base_width, base_width, base_width*2, base_width*2],
                 'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                               for i in range(3, 8)}}

    return arch


class Generator(nn.Module):
    def __init__(self, G_ch=64, G_depth=2, bottom_width=4, resolution=128,
                 G_kernel_size=3, G_attn='64',
                 num_G_SVs=1, num_G_SV_itrs=1, hier=False,
                 cross_replica=False, mybn=False,
                 G_activation=nn.ReLU(inplace=False),
                 BN_eps=1e-5, SN_eps=1e-12,
                 G_init='ortho', skip_init=False,
                 G_param='SN', norm_style='bn'):
        super(Generator, self).__init__()
        # Channel width multiplier
        self.ch = G_ch
        # Number of resblocks per stage
        self.G_depth = G_depth
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[resolution]

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)

        self.which_bn = functools.partial(layers.bn,
                                          cross_replica=self.cross_replica,
                                          mybn=self.mybn,
                                          norm_style=self.norm_style,
                                          eps=self.BN_eps)

        # Prepare model
        # First conv layer to project into feature-space
        self.initial_conv = nn.Sequential(self.which_conv(3, self.arch['in_channels'][0]),
                                          layers.bn(self.arch['in_channels'][0],
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn),
                                          self.activation
                                          )

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[GBlock(in_channels=self.arch['in_channels'][index],
                                    out_channels=self.arch['in_channels'][index] if g_index == 0 else
                                    self.arch['out_channels'][index],
                                    which_conv=self.which_conv,
                                    which_bn=self.which_bn,
                                    activation=self.activation,
                                    upsample=(functools.partial(F.interpolate, scale_factor=2)
                                              if self.arch['upsample'][index] and g_index == (
                                                self.G_depth - 1) else None))]
                            for g_index in range(self.G_depth)]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn),
                                          self.activation,
                                          self.which_conv(self.arch['out_channels'][-1], 3))

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    def forward(self, z):
        # First conv layer to convert into correct filter-space.
        h = self.initial_conv(z)
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h)

        # Apply batchnorm-relu-conv-tanh at output
        return (torch.tanh(self.output_layer(h)), )

def biggan_medium(num_filters):
    return Generator(num_filters)