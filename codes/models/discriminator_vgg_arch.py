import torch
import torch.nn as nn

from models.arch_util import ConvBnLelu, ConvGnLelu, ExpansionBlock, ConvGnSilu, ResidualBlockGN
import torch.nn.functional as F

from trainer.networks import register_model
from utils.util import checkpoint, opt_get


class Discriminator_VGG_128(nn.Module):
    # input_img_factor = multiplier to support images over 128x128. Only certain factors are supported.
    def __init__(self, in_nc, nf, input_img_factor=1, extra_conv=False):
        super(Discriminator_VGG_128, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
        final_nf = nf * 8

        self.extra_conv = extra_conv
        if self.extra_conv:
            self.conv5_0 = nn.Conv2d(nf * 8, nf * 16, 3, 1, 1, bias=False)
            self.bn5_0 = nn.BatchNorm2d(nf * 16, affine=True)
            self.conv5_1 = nn.Conv2d(nf * 16, nf * 16, 4, 2, 1, bias=False)
            self.bn5_1 = nn.BatchNorm2d(nf * 16, affine=True)
            input_img_factor = input_img_factor // 2
            final_nf = nf * 16

        self.linear1 = nn.Linear(final_nf * 4 * input_img_factor * 4 * input_img_factor, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        #fea = torch.cat([fea, skip_med], dim=1)
        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        #fea = torch.cat([fea, skip_lo], dim=1)
        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        if self.extra_conv:
            fea = self.lrelu(self.bn5_0(self.conv5_0(fea)))
            fea = self.lrelu(self.bn5_1(self.conv5_1(fea)))

        fea = fea.contiguous().view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


@register_model
def register_discriminator_vgg_128(opt_net, opt):
    return Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'], input_img_factor=opt_net['image_size'] / 128,
                                 extra_conv=opt_net['extra_conv'])


class Discriminator_VGG_128_GN(nn.Module):
    # input_img_factor = multiplier to support images over 128x128. Only certain factors are supported.
    def __init__(self, in_nc, nf, input_img_factor=1, do_checkpointing=False, extra_conv=False):
        super(Discriminator_VGG_128_GN, self).__init__()
        self.do_checkpointing = do_checkpointing

        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.GroupNorm(8, nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.GroupNorm(8, nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.GroupNorm(8, nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.GroupNorm(8, nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.GroupNorm(8, nf * 4, affine=True)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.GroupNorm(8, nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.GroupNorm(8, nf * 8, affine=True)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.GroupNorm(8, nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.GroupNorm(8, nf * 8, affine=True)

        self.extra_conv = extra_conv
        if extra_conv:
            self.conv5_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
            self.bn5_0 = nn.GroupNorm(8, nf * 8, affine=True)
            self.conv5_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
            self.bn5_1 = nn.GroupNorm(8, nf * 8, affine=True)
            input_img_factor = input_img_factor / 2
        final_nf = nf * 8

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.linear1 = nn.Linear(int(final_nf * 4 * input_img_factor * 4 * input_img_factor), 100)
        self.linear2 = nn.Linear(100, 1)

    def compute_body(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        #fea = torch.cat([fea, skip_med], dim=1)
        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        #fea = torch.cat([fea, skip_lo], dim=1)
        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        if self.extra_conv:
            fea = self.lrelu(self.bn5_0(self.conv5_0(fea)))
            fea = self.lrelu(self.bn5_1(self.conv5_1(fea)))
        return fea

    def forward(self, x):
        if self.do_checkpointing:
            fea = checkpoint(self.compute_body, x)
        else:
            fea = self.compute_body(x)
        fea = fea.contiguous().view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


@register_model
def register_discriminator_vgg_128(opt_net, opt):
    return Discriminator_VGG_128_GN(in_nc=opt_net['in_nc'], nf=opt_net['nf'],
                                    input_img_factor=opt_net['image_size'] / 128,
                                    extra_conv=opt_get(opt_net, ['extra_conv'], False),
                                    do_checkpointing=opt_get(opt_net, ['do_checkpointing'], False))


class DiscriminatorVGG448GN(nn.Module):
    # input_img_factor = multiplier to support images over 128x128. Only certain factors are supported.
    def __init__(self, in_nc, nf, do_checkpointing=False):
        super().__init__()
        self.do_checkpointing = do_checkpointing

        # 448x448
        self.convn1_0 = nn.Conv2d(in_nc, nf // 2, 3, 1, 1, bias=True)
        self.convn1_1 = nn.Conv2d(nf // 2, nf // 2, 4, 2, 1, bias=False)
        self.bnn1_1 = nn.GroupNorm(8, nf // 2, affine=True)

        # 224x224 (new head)
        self.conv0_0_new = nn.Conv2d(nf // 2, nf, 3, 1, 1, bias=False)
        self.bn0_0 = nn.GroupNorm(8, nf, affine=True)
        # 224x224 (old head)
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)  # Unused.
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.GroupNorm(8, nf, affine=True)
        # 112x112
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.GroupNorm(8, nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.GroupNorm(8, nf * 2, affine=True)
        # 56x56
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.GroupNorm(8, nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.GroupNorm(8, nf * 4, affine=True)
        # 28x28
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.GroupNorm(8, nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.GroupNorm(8, nf * 8, affine=True)
        # 14x14
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.GroupNorm(8, nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.GroupNorm(8, nf * 8, affine=True)

        # out: 7x7

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        final_nf = nf * 8
        self.linear1 = nn.Linear(int(final_nf * 7 * 7), 100)
        self.linear2 = nn.Linear(100, 1)

        # Assign all new heads to the new param group.2
        for m in [self.convn1_0, self.convn1_1, self.bnn1_1, self.conv0_0_new, self.bn0_0]:
            for p in m.parameters():
                p.PARAM_GROUP = 'new_head'

    def compute_body(self, x):
        fea = self.lrelu(self.convn1_0(x))
        fea = self.lrelu(self.bnn1_1(self.convn1_1(fea)))

        fea = self.lrelu(self.bn0_0(self.conv0_0_new(fea)))
        # fea = self.lrelu(self.conv0_0(x)) <- replaced
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
        return fea

    def forward(self, x):
        if self.do_checkpointing:
            fea = checkpoint(self.compute_body, x)
        else:
            fea = self.compute_body(x)
        fea = fea.contiguous().view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


@register_model
def register_discriminator_vgg_448(opt_net, opt):
    return DiscriminatorVGG448GN(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
