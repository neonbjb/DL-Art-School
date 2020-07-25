import torch
import torch.nn as nn
import torchvision
from models.archs.arch_util import ConvBnLelu, ConvGnLelu, ExpansionBlock, ConvGnSilu
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


import functools
from models.archs.SwitchedResidualGenerator_arch import MultiConvBlock, ConfigurableSwitchComputer, BareConvSwitch
from switched_conv_util import save_attention_to_image
from switched_conv import compute_attention_specificity, AttentionNorm


class ReducingMultiplexer(nn.Module):
    def __init__(self, nf, num_channels):
        super(ReducingMultiplexer, self).__init__()
        self.conv1_0 = ConvGnSilu(nf, nf * 2, kernel_size=3, bias=False)
        self.conv1_1 = ConvGnSilu(nf * 2, nf * 2, kernel_size=3, stride=2, bias=False)
        # [128, 32, 32]
        self.conv2_0 = ConvGnSilu(nf * 2, nf * 4, kernel_size=3, bias=False)
        self.conv2_1 = ConvGnSilu(nf * 4, nf * 4, kernel_size=3, stride=2, bias=False)
        # [256, 16, 16]
        self.conv3_0 = ConvGnSilu(nf * 4, nf * 8, kernel_size=3, bias=False)
        self.conv3_1 = ConvGnSilu(nf * 8, nf * 8, kernel_size=3, stride=2, bias=False)
        self.exp1 = ExpansionBlock(nf * 8, nf * 4)
        self.exp2 = ExpansionBlock(nf * 4, nf * 2)
        self.exp3 = ExpansionBlock(nf * 2, nf)
        self.collapse = ConvGnSilu(nf, num_channels, norm=False, bias=True)

    def forward(self, x):
        fea1 = self.conv1_0(x)
        fea1 = self.conv1_1(fea1)
        fea2 = self.conv2_0(fea1)
        fea2 = self.conv2_1(fea2)
        fea3 = self.conv3_0(fea2)
        fea3 = self.conv3_1(fea3)
        up = self.exp1(fea3, fea2)
        up = self.exp2(up, fea1)
        up = self.exp3(up, x)
        return self.collapse(up)


# Differs from ConfigurableSwitchComputer in that the connections are not residual and the multiplexer is fed directly in.
class ConfigurableLinearSwitchComputer(nn.Module):
    def __init__(self, out_filters, multiplexer_net, pre_transform_block, transform_block, transform_count, attention_norm,
                 init_temp=20, add_scalable_noise_to_transforms=False):
        super(ConfigurableLinearSwitchComputer, self).__init__()

        self.multiplexer = multiplexer_net
        self.pre_transform = pre_transform_block
        self.transforms = nn.ModuleList([transform_block() for _ in range(transform_count)])
        self.add_noise = add_scalable_noise_to_transforms
        self.noise_scale = nn.Parameter(torch.full((1,), float(1e-3)))

        # And the switch itself, including learned scalars
        self.switch = BareConvSwitch(initial_temperature=init_temp, attention_norm=AttentionNorm(transform_count, accumulator_size=16 * transform_count) if attention_norm else None)
        self.post_switch_conv = ConvBnLelu(out_filters, out_filters, norm=False, bias=True)
        # The post_switch_conv gets a low scale initially. The network can decide to magnify it (or not)
        # depending on its needs.
        self.psc_scale = nn.Parameter(torch.full((1,), float(.1)))

    def forward(self, x, output_attention_weights=False, extra_arg=None):
        if self.add_noise:
            rand_feature = torch.randn_like(x) * self.noise_scale
            x = x + rand_feature

        if self.pre_transform:
            x = self.pre_transform(x)
        xformed = [t.forward(x) for t in self.transforms]
        m = self.multiplexer(x)

        outputs, attention = self.switch(xformed, m, True)
        outputs = self.post_switch_conv(outputs)
        if output_attention_weights:
            return outputs, attention
        else:
            return outputs

    def set_temperature(self, temp):
        self.switch.set_attention_temperature(temp)


def create_switched_downsampler(nf, nf_out, num_channels, initial_temp=10):
    multiplx = ReducingMultiplexer(nf, num_channels)
    pretransform = None
    transform_fn = functools.partial(MultiConvBlock, nf, nf, nf_out, kernel_size=3, depth=2)
    return ConfigurableLinearSwitchComputer(nf_out, multiplx,
                                       pre_transform_block=pretransform, transform_block=transform_fn,
                                       attention_norm=True,
                                       transform_count=num_channels, init_temp=initial_temp,
                                       add_scalable_noise_to_transforms=False)


class Discriminator_switched(nn.Module):
    def __init__(self, in_nc, nf, initial_temp=10, final_temperature_step=50000):
        super(Discriminator_switched, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = ConvGnLelu(in_nc, nf, kernel_size=3, bias=True, activation=False)
        self.conv0_1 = ConvGnLelu(nf, nf, kernel_size=3, stride=2, bias=False)
        # [64, 64, 64]
        self.sw = create_switched_downsampler(nf, nf, 8)
        self.switches = [self.sw]
        self.conv1_1 = ConvGnLelu(nf, nf * 2, kernel_size=3, stride=2, bias=False)
        # [128, 32, 32]
        self.conv2_0 = ConvGnLelu(nf * 2, nf * 4, kernel_size=3, bias=False)
        self.conv2_1 = ConvGnLelu(nf * 4, nf * 4, kernel_size=3, stride=2, bias=False)
        # [256, 16, 16]
        self.conv3_0 = ConvGnLelu(nf * 4, nf * 8, kernel_size=3, bias=False)
        self.conv3_1 = ConvGnLelu(nf * 8, nf * 8, kernel_size=3, stride=2, bias=False)
        # [512, 8, 8]
        self.conv4_0 = ConvGnLelu(nf * 8, nf * 8, kernel_size=3, bias=False)
        self.conv4_1 = ConvGnLelu(nf * 8, nf * 8, kernel_size=3, stride=2, bias=False)

        self.exp1 = ExpansionBlock(nf * 8, nf * 8, block=ConvGnLelu)
        self.exp2 = ExpansionBlock(nf * 8, nf * 4, block=ConvGnLelu)
        self.exp3 = ExpansionBlock(nf * 4, nf * 2, block=ConvGnLelu)
        self.proc3 = ConvGnLelu(nf * 2, nf * 2, bias=False)
        self.collapse3 = ConvGnLelu(nf * 2, 1, bias=True, norm=False, activation=False)

        self.init_temperature = initial_temp
        self.final_temperature_step = final_temperature_step
        self.attentions = None

    def forward(self, x, flatten=True):
        fea0 = self.conv0_0(x)
        fea0 = self.conv0_1(fea0)

        fea1, att = self.sw(fea0, True)
        self.attentions = [att]
        fea1 = self.conv1_1(fea1)

        fea2 = self.conv2_0(fea1)
        fea2 = self.conv2_1(fea2)

        fea3 = self.conv3_0(fea2)
        fea3 = self.conv3_1(fea3)

        fea4 = self.conv4_0(fea3)
        fea4 = self.conv4_1(fea4)

        u1 = self.exp1(fea4, fea3)
        u2 = self.exp2(u1, fea2)
        u3 = self.exp3(u2, fea1)

        loss3 = self.collapse3(self.proc3(u3))
        return loss3.view(-1, 1)

    def pixgan_parameters(self):
        return 1, 4

    def set_temperature(self, temp):
        [sw.set_temperature(temp) for sw in self.switches]

    def update_for_step(self, step, experiments_path='.'):
        if self.attentions:
            for i, sw in enumerate(self.switches):
                temp_loss_per_step = (self.init_temperature - 1) / self.final_temperature_step
                sw.set_temperature(min(self.init_temperature,
                                       max(self.init_temperature - temp_loss_per_step * step, 1)))
            if step % 50 == 0:
                [save_attention_to_image(experiments_path, self.attentions[i], 8, step, "disc_a%i" % (i+1,), l_mult=10) for i in range(len(self.attentions))]

    def get_debug_values(self, step):
        temp = self.switches[0].switch.temperature
        mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
        means = [i[0] for i in mean_hists]
        hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
        val = {"disc_switch_temperature": temp}
        for i in range(len(means)):
            val["disc_switch_%i_specificity" % (i,)] = means[i]
            val["disc_switch_%i_histogram" % (i,)] = hists[i]
        return val


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