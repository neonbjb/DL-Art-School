import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torchvision


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
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

# 5-channel residual block that uses attention in the convolutions.
class AttentiveResidualDenseBlock_5C(ResidualDenseBlock_5C):
    def __init__(self, nf=64, gc=32, num_convs=8, init_temperature=1):
        super(AttentiveResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = arch_util.DynamicConv2d(nf, gc, 3, 1, 1, bias=bias, num_convs=num_convs,
                                             initial_temperature=init_temperature)
        self.conv2 = arch_util.DynamicConv2d(nf + gc, gc, 3, 1, 1, bias=bias, num_convs=num_convs,
                                             initial_temperature=init_temperature)
        self.conv3 = arch_util.DynamicConv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias, num_convs=num_convs,
                                             initial_temperature=init_temperature)
        self.conv4 = arch_util.DynamicConv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias, num_convs=num_convs,
                                             initial_temperature=init_temperature)
        self.conv5 = arch_util.DynamicConv2d(nf + 4 * gc, gc, 3, 1, 1, bias=bias, num_convs=num_convs,
                                             initial_temperature=init_temperature)

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
                                     0.1)

    def set_temperature(self, temp):
        self.conv1.set_attention_temperature(temp)
        self.conv2.set_attention_temperature(temp)
        self.conv3.set_attention_temperature(temp)
        self.conv4.set_attention_temperature(temp)
        self.conv5.set_attention_temperature(temp)


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class AttentiveRRDB(RRDB):
    def __init__(self, nf, gc=32, num_convs=8, init_temperature=1):
        super(RRDB, self).__init__()
        self.RDB1 = AttentiveResidualDenseBlock_5C(nf, gc, num_convs, init_temperature)
        self.RDB2 = AttentiveResidualDenseBlock_5C(nf, gc, num_convs, init_temperature)
        self.RDB3 = AttentiveResidualDenseBlock_5C(nf, gc, num_convs, init_temperature)

    def set_temperature(self, temp):
        self.RDB1.set_temperature(temp)
        self.RDB2.set_temperature(temp)
        self.RDB3.set_temperature(temp)

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=2, initial_stride=1,
                 rrdb_block_f=None):
        super(RRDBNet, self).__init__()
        if rrdb_block_f is None:
            rrdb_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 7, initial_stride, padding=3, bias=True)
        self.RRDB_trunk, self.rrdb_layers = arch_util.make_layer(rrdb_block_f, nb, True)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # Sets the softmax temperature of each RRDB layer. Only works if you are using attentive
    # convolutions.
    def set_temperature(self, temp):
        for layer in self.rrdb_layers:
            layer.set_temperature(temp)

    def forward(self, x):
        fea = self.conv_first(x)
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

# Variant of RRDBNet that is "assisted" by an external pretrained image classifier whose
# intermediate layers have been splayed out, pixel-shuffled, and fed back in.
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
        assist_nf = [2, 4, 8, 16]  # Fixed for resnet. Re-evaluate if using other networks.
        self.assist1 = RRDB(nf + assist_nf[0], gc)
        self.assist2 = RRDB(nf + sum(assist_nf[:2]), gc)
        self.assist3 = RRDB(nf + sum(assist_nf[:3]), gc)
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
        x = self.assistnet.layer4(x)
        l4 = F.pixel_shuffle(x, 32)
        return l1, l2, l3, l4

    def forward(self, x):
        # Invoke the assistant net first.
        l1, l2, l3, l4 = self.res_extract(x)

        fea = self.conv_first(x)
        fea = self.assist1(torch.cat([fea, l4], dim=1))
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