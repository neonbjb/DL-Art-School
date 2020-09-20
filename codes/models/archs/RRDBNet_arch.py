import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
from models.archs.arch_util import PixelUnshuffle
import torchvision
from torch.utils.checkpoint import checkpoint


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True, late_stage_kernel_size=3, late_stage_padding=1):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, late_stage_kernel_size, 1, late_stage_padding, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, late_stage_kernel_size, 1, late_stage_padding, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, late_stage_kernel_size, 1, late_stage_padding, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
                                     0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = checkpoint(self.RDB1, x)
        out = checkpoint(self.RDB2, out)
        out = checkpoint(self.RDB3, out)
        return out * 0.2 + x


class LowDimRRDB(RRDB):
    def __init__(self, nf, gc=32, dimensional_adjustment=4):
        super(LowDimRRDB, self).__init__(nf * (dimensional_adjustment ** 2), gc * (dimensional_adjustment ** 2))
        self.unshuffle = PixelUnshuffle(dimensional_adjustment)
        self.shuffle = nn.PixelShuffle(dimensional_adjustment)

    def forward(self, x):
        x = self.unshuffle(x)
        x = super(LowDimRRDB, self).forward(x)
        return self.shuffle(x)


# Identical to LowDimRRDB but wraps an RRDB rather than inheriting from it. TODO: remove LowDimRRDB when backwards
# compatibility is no longer desired.
class LowDimRRDBWrapper(nn.Module):
    # Do not specify nf or gc on the partial_rrdb passed in. That will be done by the wrapper.
    def __init__(self, nf, partial_rrdb, gc=32, dimensional_adjustment=4):
        super(LowDimRRDBWrapper, self).__init__()
        self.rrdb = partial_rrdb(nf=nf * (dimensional_adjustment ** 2), gc=gc * (dimensional_adjustment ** 2))
        self.unshuffle = PixelUnshuffle(dimensional_adjustment)
        self.shuffle = nn.PixelShuffle(dimensional_adjustment)

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.rrdb(x)
        return self.shuffle(x)


# This module performs the majority of the processing done by RRDBNet. It just doesn't have the upsampling at the end.
class RRDBTrunk(nn.Module):
    def __init__(self, nf_in, nf_out, nb, gc=32, initial_stride=1, rrdb_block_f=None, conv_first_block=None):
        super(RRDBTrunk, self).__init__()
        if rrdb_block_f is None:
            rrdb_block_f = functools.partial(RRDB, nf=nf_out, gc=gc)

        if conv_first_block is None:
            self.conv_first = nn.Conv2d(nf_in, nf_out, 7, initial_stride, padding=3, bias=True)
        else:
            self.conv_first = conv_first_block

        self.RRDB_trunk, self.rrdb_layers = arch_util.make_layer(rrdb_block_f, nb, True)
        self.trunk_conv = nn.Conv2d(nf_out, nf_out, 3, 1, 1, bias=True)

    # Sets the softmax temperature of each RRDB layer. Only works if you are using attentive
    # convolutions.
    def set_temperature(self, temp):
        for layer in self.rrdb_layers:
            layer.set_temperature(temp)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        return fea


# Adds some base methods that all RRDB* classes will use.
class RRDBBase(nn.Module):
    def __init__(self):
        super(RRDBBase, self).__init__()

    # Sets the softmax temperature of each RRDB layer. Only works if you are using attentive
    # convolutions.
    def set_temperature(self, temp):
        for trunk in self.trunks:
            for layer in trunk.rrdb_layers:
                layer.set_temperature(temp)


# This class uses a RRDBTrunk to perform processing on an image, then upsamples it.
class RRDBNet(RRDBBase):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=2, initial_stride=1,
                 rrdb_block_f=None):
        super(RRDBNet, self).__init__()

        # Trunk - does actual processing.
        self.trunk = RRDBTrunk(in_nc, nf, nb, gc, initial_stride, rrdb_block_f)
        self.trunks = [self.trunk]

        # Upsampling
        self.scale = scale
        self.upconv1 = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.trunk(x)

        if self.scale >= 2:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = self.lrelu(self.upconv1(fea))
        if self.scale >= 4:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = self.lrelu(self.upconv2(fea))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

    def load_state_dict(self, state_dict, strict=True):
        # The parameters in self.trunk used to be in this class. To support loading legacy saves, restore them.
        t_state = self.trunk.state_dict()
        for k in t_state.keys():
            if k in state_dict.keys():
                state_dict["trunk.%s" % (k,)] = state_dict.pop(k)
        super(RRDBNet, self).load_state_dict(state_dict, strict)


# Variant of RRDBNet that is "assisted" by an external pretrained image classifier whose
# intermediate layers have been splayed out, pixel-shuffled, and fed back in.
# TODO: Convert to use new RRDBBase hierarchy.
class AssistedRRDBNet(nn.Module):
    # in_nc=number of input channels.
    # out_nc=number of output channels.
    # nf=internal filter count
    # nb=number of additional blocks after the assistance layers.
    # gc=growth channel inside of residual blocks
    # scale=the number of times the output is doubled in size.
    # initial_stride=the stride on the first conv. can be used to downsample the image for processing.
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=2, initial_stride=1):
        super(AssistedRRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 7, initial_stride, padding=3, bias=True)

        # Set-up the assist-net, which should do feature extraction for us.
        self.assistnet = torchvision.models.wide_resnet50_2(pretrained=True)
        self.set_enable_assistnet_training(False)
        assist_nf = [4, 8, 16]  # Fixed for resnet. Re-evaluate if using other networks.
        self.assist2 = RRDB(nf + assist_nf[0], gc)
        self.assist3 = RRDB(nf + sum(assist_nf[:2]), gc)
        self.assist4 = RRDB(nf + sum(assist_nf), gc)
        nf = nf + sum(assist_nf)

        # After this, it's just a "standard" RRDB net.
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.RRDB_trunk = arch_util.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def set_enable_assistnet_training(self, en):
        for p in self.assistnet.parameters():
            p.requires_grad = en

    def res_extract(self, x):
        # Width and height must be factors of 16 to use this architecture. Check that here.
        (b, f, w, h) = x.shape
        assert w % 16 == 0
        assert h % 16 == 0

        x = self.assistnet.conv1(x)
        x = self.assistnet.bn1(x)
        x = self.assistnet.relu(x)
        x = self.assistnet.maxpool(x)

        x = self.assistnet.layer1(x)
        l1 = F.pixel_shuffle(x, 4)
        x = self.assistnet.layer2(x)
        l2 = F.pixel_shuffle(x, 8)
        x = self.assistnet.layer3(x)
        l3 = F.pixel_shuffle(x, 16)
        return l1, l2, l3

    def forward(self, x):
        # Invoke the assistant net first.
        l1, l2, l3 = self.res_extract(x)

        fea = self.conv_first(x)
        fea = self.assist2(torch.cat([fea, l3], dim=1))
        fea = self.assist3(torch.cat([fea, l2], dim=1))
        fea = self.assist4(torch.cat([fea, l1], dim=1))

        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        if self.scale >= 2:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = self.lrelu(self.upconv1(fea))
        if self.scale >= 4:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = self.lrelu(self.upconv2(fea))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return (out,)


class PixShuffleInitialConv(nn.Module):
    def __init__(self, reduction_factor, nf_out):
        super(PixShuffleInitialConv, self).__init__()
        self.conv = nn.Conv2d(3 * (reduction_factor ** 2), nf_out, 1)
        self.unshuffle = PixelUnshuffle(reduction_factor)

    def forward(self, x):
        (b, f, w, h) = x.shape
        # This module can only be applied to input images (with 3 channels)
        assert f == 3

        x = self.unshuffle(x)
        return self.conv(x)


# This class uses a RRDBTrunk to perform processing on an image, then upsamples it.
class PixShuffleRRDB(RRDBBase):
    def __init__(self, nf, nb, gc=32, scale=2, rrdb_block_f=None):
        super(PixShuffleRRDB, self).__init__()

        # This class does a 4x pixel shuffle on the filter count inside the trunk, so nf must be divisible by 16.
        assert nf % 16 == 0

        # Trunk - does actual processing.
        self.trunk = RRDBTrunk(3, nf, nb, gc, 1, rrdb_block_f, PixShuffleInitialConv(4, nf))
        self.trunks = [self.trunk]

        # Upsampling
        pix_nf = int(nf/16)
        self.scale = scale
        self.upconv1 = nn.Conv2d(pix_nf, pix_nf, 5, 1, padding=2, bias=True)
        self.upconv2 = nn.Conv2d(pix_nf, pix_nf, 5, 1, padding=2, bias=True)
        self.HRconv = nn.Conv2d(pix_nf, pix_nf, 5, 1, padding=2, bias=True)
        self.conv_last = nn.Conv2d(pix_nf, 3, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.trunk(x)
        fea = self.pixel_shuffle(fea)

        if self.scale >= 2:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = self.lrelu(self.upconv1(fea))
        if self.scale >= 4:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = self.lrelu(self.upconv2(fea))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return (out,)