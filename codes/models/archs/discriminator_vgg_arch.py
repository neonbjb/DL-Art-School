import torch
import torch.nn as nn
import torchvision
from models.archs.arch_util import ConvBnLelu, ConvGnLelu, ExpansionBlock
import torch.nn.functional as F


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


class Discriminator_VGG_PixLoss(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_PixLoss, self).__init__()
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

        self.reduce_1 = ConvGnLelu(nf * 8, nf * 4, bias=False)
        self.pix_loss_collapse = ConvGnLelu(nf * 4, 1, bias=False, norm=False, activation=False)

        # Pyramid network: upsample with residuals and produce losses at multiple resolutions.
        self.up3_decimate = ConvGnLelu(nf * 8, nf * 8, kernel_size=3, bias=True, activation=False)
        self.up3_converge = ConvGnLelu(nf * 16, nf * 8, kernel_size=3, bias=False)
        self.up3_proc = ConvGnLelu(nf * 8, nf * 8, bias=False)
        self.up3_reduce = ConvGnLelu(nf * 8, nf * 4, bias=False)
        self.up3_pix = ConvGnLelu(nf * 4, 1, bias=False, norm=False, activation=False)

        self.up2_decimate = ConvGnLelu(nf * 8, nf * 4, kernel_size=1, bias=True, activation=False)
        self.up2_converge = ConvGnLelu(nf * 8, nf * 4, kernel_size=3, bias=False)
        self.up2_proc = ConvGnLelu(nf * 4, nf * 4, bias=False)
        self.up2_reduce = ConvGnLelu(nf * 4, nf * 2, bias=False)
        self.up2_pix = ConvGnLelu(nf * 2, 1, bias=False, norm=False, activation=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, flatten=True):
        fea0 = self.lrelu(self.conv0_0(x))
        fea0 = self.lrelu(self.bn0_1(self.conv0_1(fea0)))

        fea1 = self.lrelu(self.bn1_0(self.conv1_0(fea0)))
        fea1 = self.lrelu(self.bn1_1(self.conv1_1(fea1)))

        fea2 = self.lrelu(self.bn2_0(self.conv2_0(fea1)))
        fea2 = self.lrelu(self.bn2_1(self.conv2_1(fea2)))

        fea3 = self.lrelu(self.bn3_0(self.conv3_0(fea2)))
        fea3 = self.lrelu(self.bn3_1(self.conv3_1(fea3)))

        fea4 = self.lrelu(self.bn4_0(self.conv4_0(fea3)))
        fea4 = self.lrelu(self.bn4_1(self.conv4_1(fea4)))

        loss = self.reduce_1(fea4)
        # "Weight" all losses the same by interpolating them to the highest dimension.
        loss = self.pix_loss_collapse(loss)
        loss = F.interpolate(loss, scale_factor=4, mode="nearest")

        # And the pyramid network!
        dec3 = self.up3_decimate(F.interpolate(fea4, scale_factor=2, mode="nearest"))
        dec3 = torch.cat([dec3, fea3], dim=1)
        dec3 = self.up3_converge(dec3)
        dec3 = self.up3_proc(dec3)
        loss3 = self.up3_reduce(dec3)
        loss3 = self.up3_pix(loss3)
        loss3 = F.interpolate(loss3, scale_factor=2, mode="nearest")

        dec2 = self.up2_decimate(F.interpolate(dec3, scale_factor=2, mode="nearest"))
        dec2 = torch.cat([dec2, fea2], dim=1)
        dec2 = self.up2_converge(dec2)
        dec2 = self.up2_proc(dec2)
        dec2 = self.up2_reduce(dec2)
        loss2 = self.up2_pix(dec2)

        # Compress all of the loss values into the batch dimension. The actual loss attached to this output will
        # then know how to handle them.
        combined_losses = torch.cat([loss, loss3, loss2], dim=1)
        return combined_losses.view(-1, 1)

    def pixgan_parameters(self):
        return 3, 8


class Discriminator_UNet(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_UNet, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = ConvGnLelu(in_nc, nf, kernel_size=3, bias=True, activation=False)
        self.conv0_1 = ConvGnLelu(nf, nf, kernel_size=3, stride=2, bias=False)
        # [64, 64, 64]
        self.conv1_0 = ConvGnLelu(nf, nf * 2, kernel_size=3, bias=False)
        self.conv1_1 = ConvGnLelu(nf * 2, nf * 2, kernel_size=3, stride=2, bias=False)
        # [128, 32, 32]
        self.conv2_0 = ConvGnLelu(nf * 2, nf * 4, kernel_size=3, bias=False)
        self.conv2_1 = ConvGnLelu(nf * 4, nf * 4, kernel_size=3, stride=2, bias=False)
        # [256, 16, 16]
        self.conv3_0 = ConvGnLelu(nf * 4, nf * 8, kernel_size=3, bias=False)
        self.conv3_1 = ConvGnLelu(nf * 8, nf * 8, kernel_size=3, stride=2, bias=False)
        # [512, 8, 8]
        self.conv4_0 = ConvGnLelu(nf * 8, nf * 8, kernel_size=3, bias=False)
        self.conv4_1 = ConvGnLelu(nf * 8, nf * 8, kernel_size=3, stride=2, bias=False)

        self.up1 = ExpansionBlock(nf * 8, nf * 8, block=ConvGnLelu)
        self.proc1 = ConvGnLelu(nf * 8, nf * 8, bias=False)
        self.collapse1 = ConvGnLelu(nf * 8, 1, bias=True, norm=False, activation=False)

        self.up2 = ExpansionBlock(nf * 8, nf * 4, block=ConvGnLelu)
        self.proc2 = ConvGnLelu(nf * 4, nf * 4, bias=False)
        self.collapse2 = ConvGnLelu(nf * 4, 1, bias=True, norm=False, activation=False)

        self.up3 = ExpansionBlock(nf * 4, nf * 2, block=ConvGnLelu)
        self.proc3 = ConvGnLelu(nf * 2, nf * 2, bias=False)
        self.collapse3 = ConvGnLelu(nf * 2, 1, bias=True, norm=False, activation=False)

    def forward(self, x, flatten=True):
        fea0 = self.conv0_0(x)
        fea0 = self.conv0_1(fea0)

        fea1 = self.conv1_0(fea0)
        fea1 = self.conv1_1(fea1)

        fea2 = self.conv2_0(fea1)
        fea2 = self.conv2_1(fea2)

        fea3 = self.conv3_0(fea2)
        fea3 = self.conv3_1(fea3)

        fea4 = self.conv4_0(fea3)
        fea4 = self.conv4_1(fea4)

        # And the pyramid network!
        u1 = self.up1(fea4, fea3)
        loss1 = self.collapse1(self.proc1(u1))
        u2 = self.up2(u1, fea2)
        loss2 = self.collapse2(self.proc2(u2))
        u3 = self.up3(u2, fea1)
        loss3 = self.collapse3(self.proc3(u3))
        res = loss3.shape[2:]

        # Compress all of the loss values into the batch dimension. The actual loss attached to this output will
        # then know how to handle them.
        combined_losses = torch.cat([F.interpolate(loss1, scale_factor=4),
                                     F.interpolate(loss2, scale_factor=2),
                                     F.interpolate(loss3, scale_factor=1)], dim=1)
        return combined_losses.view(-1, 1)

    def pixgan_parameters(self):
        return 3, 4

