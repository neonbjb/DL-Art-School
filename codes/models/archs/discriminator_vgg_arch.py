import torch
import torch.nn as nn

from models.archs.RRDBNet_arch import RRDB, RRDBWithBypass
from models.archs.arch_util import ConvBnLelu, ConvGnLelu, ExpansionBlock, ConvGnSilu, ResidualBlockGN
import torch.nn.functional as F
from models.archs.SwitchedResidualGenerator_arch import gather_2d
from utils.util import checkpoint


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


class CrossCompareBlock(nn.Module):
    def __init__(self, nf_in, nf_out):
        super(CrossCompareBlock, self).__init__()
        self.conv_hr_merge = ConvGnLelu(nf_in * 2, nf_in, kernel_size=1, bias=False, activation=False, norm=True)
        self.proc_hr = ConvGnLelu(nf_in, nf_out, kernel_size=3, bias=False, activation=True, norm=True)
        self.proc_lr = ConvGnLelu(nf_in, nf_out, kernel_size=3, bias=False, activation=True, norm=True)
        self.reduce_hr = ConvGnLelu(nf_out, nf_out, kernel_size=3, stride=2, bias=False, activation=True, norm=True)
        self.reduce_lr = ConvGnLelu(nf_out, nf_out, kernel_size=3, stride=2, bias=False, activation=True, norm=True)

    def forward(self, hr, lr):
        hr = self.conv_hr_merge(torch.cat([hr, lr], dim=1))
        hr = self.proc_hr(hr)
        hr = self.reduce_hr(hr)

        lr = self.proc_lr(lr)
        lr = self.reduce_lr(lr)

        return hr, lr


class CrossCompareDiscriminator(nn.Module):
    def __init__(self, in_nc, ref_channels, nf, scale=4):
        super(CrossCompareDiscriminator, self).__init__()
        assert scale == 2 or scale == 4

        self.init_conv_hr = ConvGnLelu(in_nc, nf, stride=2, norm=False, bias=True, activation=True)
        self.init_conv_lr = ConvGnLelu(ref_channels, nf, stride=1, norm=False, bias=True, activation=True)
        if scale == 4:
            strd_2 = 2
        else:
            strd_2 = 1
        self.second_conv = ConvGnLelu(nf, nf, stride=strd_2, norm=True, bias=False, activation=True)

        self.cross1 = CrossCompareBlock(nf, nf * 2)
        self.cross2 = CrossCompareBlock(nf * 2, nf * 4)
        self.cross3 = CrossCompareBlock(nf * 4, nf * 8)
        self.cross4 = CrossCompareBlock(nf * 8, nf * 8)
        self.fproc_conv = ConvGnLelu(nf * 8, nf, norm=True, bias=True, activation=True)
        self.out_conv = ConvGnLelu(nf, 1, norm=False, bias=False, activation=False)

        self.scale = scale * 16

    def forward(self, hr, lr):
        hr = self.init_conv_hr(hr)
        hr = self.second_conv(hr)
        lr = self.init_conv_lr(lr)

        hr, lr = self.cross1(hr, lr)
        hr, lr = self.cross2(hr, lr)
        hr, lr = self.cross3(hr, lr)
        hr, _ = self.cross4(hr, lr)

        return self.out_conv(self.fproc_conv(hr)).view(-1, 1)

    # Returns tuple of (number_output_channels, scale_of_output_reduction (1/n))
    def pixgan_parameters(self):
        return 3, self.scale


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


class Discriminator_UNet_FeaOut(nn.Module):
    def __init__(self, in_nc, nf, feature_mode=False):
        super(Discriminator_UNet_FeaOut, self).__init__()
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
        self.fea_proc = ConvGnLelu(nf * 8, nf * 8, bias=True, norm=False, activation=False)
        self.collapse1 = ConvGnLelu(nf * 8, 1, bias=True, norm=False, activation=False)

        self.feature_mode = feature_mode

    def forward(self, x, output_feature_vector=False):
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
        fea_out = self.fea_proc(u1)

        combined_losses = F.interpolate(loss1, scale_factor=4)
        if output_feature_vector:
            return combined_losses.view(-1, 1), fea_out
        else:
            return combined_losses.view(-1, 1)

    def pixgan_parameters(self):
        return 1, 4


class Vgg128GnHead(nn.Module):
    def __init__(self, in_nc, nf, depth=5):
        super(Vgg128GnHead, self).__init__()
        assert depth == 4 or depth == 5  # Nothing stopping others from being implemented, just not done yet.
        self.depth = depth

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
        if depth > 4:
            # [512, 8, 8]
            self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
            self.bn4_0 = nn.GroupNorm(8, nf * 8, affine=True)
            self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
            self.bn4_1 = nn.GroupNorm(8, nf * 8, affine=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        if self.depth > 4:
            fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
            fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
        return fea


class RefDiscriminatorVgg128(nn.Module):
    # input_img_factor = multiplier to support images over 128x128. Only certain factors are supported.
    def __init__(self, in_nc, nf, input_img_factor=1):
        super(RefDiscriminatorVgg128, self).__init__()

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.feature_head = Vgg128GnHead(in_nc, nf)
        self.ref_head = Vgg128GnHead(in_nc+1, nf, depth=4)
        final_nf = nf * 8

        self.linear1 = nn.Linear(int(final_nf * 4 * input_img_factor * 4 * input_img_factor), 512)
        self.ref_linear = nn.Linear(nf * 8, 128)

        self.output_linears = nn.Sequential(
            nn.Linear(128+512, 512),
            self.lrelu,
            nn.Linear(512, 256),
            self.lrelu,
            nn.Linear(256, 128),
            self.lrelu,
            nn.Linear(128, 1)
        )

    def forward(self, x, ref, ref_center_point):
        ref = self.ref_head(ref)
        ref_center_point = ref_center_point // 16
        ref_vector = gather_2d(ref, ref_center_point)
        ref_vector = self.ref_linear(ref_vector)

        fea = self.feature_head(x)
        fea = fea.contiguous().view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))

        out = self.output_linears(torch.cat([fea, ref_vector], dim=1))
        return out


class PsnrApproximator(nn.Module):
    # input_img_factor = multiplier to support images over 128x128. Only certain factors are supported.
    def __init__(self, nf, input_img_factor=1):
        super(PsnrApproximator, self).__init__()

        # [64, 128, 128]
        self.fake_conv0_0 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.fake_conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.fake_bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.fake_conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.fake_bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.fake_conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.fake_bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.fake_conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.fake_bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.fake_conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.fake_bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)

        # [64, 128, 128]
        self.real_conv0_0 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.real_conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.real_bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.real_conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.real_bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.real_conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.real_bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.real_conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.real_bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.real_conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.real_bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)

        # [512, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
        final_nf = nf * 8

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.linear1 = nn.Linear(int(final_nf * 4 * input_img_factor * 4 * input_img_factor), 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 1)

    def compute_body1(self, real):
        fea = self.lrelu(self.real_conv0_0(real))
        fea = self.lrelu(self.real_bn0_1(self.real_conv0_1(fea)))
        fea = self.lrelu(self.real_bn1_0(self.real_conv1_0(fea)))
        fea = self.lrelu(self.real_bn1_1(self.real_conv1_1(fea)))
        fea = self.lrelu(self.real_bn2_0(self.real_conv2_0(fea)))
        fea = self.lrelu(self.real_bn2_1(self.real_conv2_1(fea)))
        return fea

    def compute_body2(self, fake):
        fea = self.lrelu(self.fake_conv0_0(fake))
        fea = self.lrelu(self.fake_bn0_1(self.fake_conv0_1(fea)))
        fea = self.lrelu(self.fake_bn1_0(self.fake_conv1_0(fea)))
        fea = self.lrelu(self.fake_bn1_1(self.fake_conv1_1(fea)))
        fea = self.lrelu(self.fake_bn2_0(self.fake_conv2_0(fea)))
        fea = self.lrelu(self.fake_bn2_1(self.fake_conv2_1(fea)))
        return fea

    def forward(self, real, fake):
        real_fea = checkpoint(self.compute_body1, real)
        fake_fea = checkpoint(self.compute_body2, fake)
        fea = torch.cat([real_fea, fake_fea], dim=1)

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))
        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = fea.contiguous().view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        fea = self.lrelu(self.linear2(fea))
        fea = self.lrelu(self.linear3(fea))
        out = self.linear4(fea)
        return out.squeeze()


class SingleImageQualityEstimator(nn.Module):
    # input_img_factor = multiplier to support images over 128x128. Only certain factors are supported.
    def __init__(self, nf, input_img_factor=1):
        super(SingleImageQualityEstimator, self).__init__()

        # [64, 128, 128]
        self.fake_conv0_0 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.fake_conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.fake_bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.fake_conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.fake_bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.fake_conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.fake_bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.fake_conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.fake_bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.fake_conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.fake_bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)

        # [512, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 2, 3, 1, 1, bias=True)
        self.conv4_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.conv4_3 = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=.2, inplace=True)

    def compute_body(self, fake):
        fea = self.lrelu(self.fake_conv0_0(fake))
        fea = self.lrelu(self.fake_bn0_1(self.fake_conv0_1(fea)))
        fea = self.lrelu(self.fake_bn1_0(self.fake_conv1_0(fea)))
        fea = self.lrelu(self.fake_bn1_1(self.fake_conv1_1(fea)))
        fea = self.lrelu(self.fake_bn2_0(self.fake_conv2_0(fea)))
        fea = self.lrelu(self.fake_bn2_1(self.fake_conv2_1(fea)))
        return fea

    def forward(self, fake):
        fea = checkpoint(self.compute_body, fake)
        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))
        fea = self.lrelu(self.conv4_0(fea))
        fea = self.lrelu(self.conv4_1(fea))
        fea = self.lrelu(self.conv4_2(fea))
        fea = self.sigmoid(self.conv4_3(fea))
        return fea
