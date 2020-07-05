import torch
from torch import nn
from switched_conv import BareConvSwitch, compute_attention_specificity
import torch.nn.functional as F
import functools
from collections import OrderedDict
from models.archs.arch_util import initialize_weights, ConvBnRelu, ConvBnLelu, ConvBnSilu
from models.archs.RRDBNet_arch import ResidualDenseBlock_5C
from models.archs.spinenet_arch import SpineNet
from switched_conv_util import save_attention_to_image


class MultiConvBlock(nn.Module):
    def __init__(self, filters_in, filters_mid, filters_out, kernel_size, depth, scale_init=1, bn=False):
        assert depth >= 2
        super(MultiConvBlock, self).__init__()
        self.noise_scale = nn.Parameter(torch.full((1,), fill_value=.01))
        self.bnconvs = nn.ModuleList([ConvBnLelu(filters_in, filters_mid, kernel_size, bn=bn, bias=False)] +
                                     [ConvBnLelu(filters_mid, filters_mid, kernel_size, bn=bn, bias=False) for i in range(depth-2)] +
                                     [ConvBnLelu(filters_mid, filters_out, kernel_size, lelu=False, bn=False, bias=False)])
        self.scale = nn.Parameter(torch.full((1,), fill_value=scale_init))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is not None:
            noise = noise * self.noise_scale
            x = x + noise
        for m in self.bnconvs:
            x = m.forward(x)
        return x * self.scale + self.bias


# VGG-style layer with Conv(stride2)->BN->Activation->Conv->BN->Activation
# Doubles the input filter count.
class HalvingProcessingBlock(nn.Module):
    def __init__(self, filters):
        super(HalvingProcessingBlock, self).__init__()
        self.bnconv1 = ConvBnLelu(filters, filters * 2, stride=2, bn=False, bias=False)
        self.bnconv2 = ConvBnLelu(filters * 2, filters * 2, bn=True, bias=False)

    def forward(self, x):
        x = self.bnconv1(x)
        return self.bnconv2(x)


# Creates a nested series of convolutional blocks. Each block processes the input data in-place and adds
# filter_growth filters. Return is (nn.Sequential, ending_filters)
def create_sequential_growing_processing_block(filters_init, filter_growth, num_convs):
    convs = []
    current_filters = filters_init
    for i in range(num_convs):
        convs.append(ConvBnSilu(current_filters, current_filters + filter_growth, bn=True, bias=False))
        current_filters += filter_growth
    return nn.Sequential(*convs), current_filters


class ConfigurableSwitchComputer(nn.Module):
    def __init__(self, base_filters, multiplexer_net, pre_transform_block, transform_block, transform_count, init_temp=20,
                 enable_negative_transforms=False, add_scalable_noise_to_transforms=False, init_scalar=1):
        super(ConfigurableSwitchComputer, self).__init__()
        self.enable_negative_transforms = enable_negative_transforms

        tc = transform_count
        if self.enable_negative_transforms:
            tc = transform_count * 2
        self.multiplexer = multiplexer_net(tc)

        self.pre_transform = pre_transform_block()
        self.transforms = nn.ModuleList([transform_block() for _ in range(transform_count)])
        self.add_noise = add_scalable_noise_to_transforms
        self.noise_scale = nn.Parameter(torch.full((1,), float(1e-3)))

        # And the switch itself, including learned scalars
        self.switch = BareConvSwitch(initial_temperature=init_temp)
        self.switch_scale = nn.Parameter(torch.full((1,), float(init_scalar)))
        self.post_switch_conv = ConvBnLelu(base_filters, base_filters, bn=False, bias=False)
        # The post_switch_conv gets a near-zero scale. The network can decide to magnify it (or not) depending on its needs.
        self.psc_scale = nn.Parameter(torch.full((1,), float(1e-3)))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, output_attention_weights=False):
        identity = x
        if self.add_noise:
            rand_feature = torch.randn_like(x) * self.noise_scale
            x = x + rand_feature

        x = self.pre_transform(x)
        xformed = [t.forward(x) for t in self.transforms]
        if self.enable_negative_transforms:
            xformed.extend([-t for t in xformed])

        m = self.multiplexer(identity)
        # Interpolate the multiplexer across the entire shape of the image.
        m = F.interpolate(m, size=xformed[0].shape[2:], mode='nearest')

        outputs, attention = self.switch(xformed, m, True)
        outputs = identity + outputs * self.switch_scale
        outputs = identity + self.post_switch_conv(outputs) * self.psc_scale
        outputs = outputs + self.bias
        if output_attention_weights:
            return outputs, attention
        else:
            return outputs

    def set_temperature(self, temp):
        self.switch.set_attention_temperature(temp)


class ConvBasisMultiplexer(nn.Module):
    def __init__(self, input_channels, base_filters, growth, reductions, processing_depth, multiplexer_channels, use_bn=True):
        super(ConvBasisMultiplexer, self).__init__()
        self.filter_conv = ConvBnSilu(input_channels, base_filters, bias=True)
        self.reduction_blocks = nn.Sequential(OrderedDict([('block%i:' % (i,), HalvingProcessingBlock(base_filters * 2 ** i)) for i in range(reductions)]))
        reduction_filters = base_filters * 2 ** reductions
        self.processing_blocks, self.output_filter_count = create_sequential_growing_processing_block(reduction_filters, growth, processing_depth)

        gap = self.output_filter_count - multiplexer_channels
        self.cbl1 = ConvBnSilu(self.output_filter_count, self.output_filter_count - (gap // 2), bn=use_bn, bias=False)
        self.cbl2 = ConvBnSilu(self.output_filter_count - (gap // 2), self.output_filter_count - (3 * gap // 4), bn=use_bn, bias=False)
        self.cbl3 = ConvBnSilu(self.output_filter_count - (3 * gap // 4), multiplexer_channels, bias=True)

    def forward(self, x):
        x = self.filter_conv(x)
        x = self.reduction_blocks(x)
        x = self.processing_blocks(x)
        x = self.cbl1(x)
        x = self.cbl2(x)
        x = self.cbl3(x)
        return x


class SpineNetMultiplexer(nn.Module):
    def __init__(self, input_channels, transform_count):
        super(SpineNetMultiplexer, self).__init__()
        self.backbone = SpineNet('49', in_channels=input_channels)
        self.rdc1 = ConvBnSilu(256, 128, kernel_size=3, bias=False)
        self.rdc2 = ConvBnSilu(128, 64, kernel_size=3, bias=False)
        self.rdc3 = ConvBnSilu(64, transform_count, bias=False, bn=False, relu=False)

    def forward(self, x):
        spine = self.backbone(x)
        feat = self.rdc1(spine[0])
        feat = self.rdc2(feat)
        feat = self.rdc3(feat)
        return feat


class ConfigurableSwitchedResidualGenerator2(nn.Module):
    def __init__(self, switch_filters, switch_growths, switch_reductions, switch_processing_layers, trans_counts, trans_kernel_sizes,
                 trans_layers, transformation_filters, initial_temp=20, final_temperature_step=50000, heightened_temp_min=1,
                 heightened_final_step=50000, upsample_factor=1, enable_negative_transforms=False,
                 add_scalable_noise_to_transforms=False):
        super(ConfigurableSwitchedResidualGenerator2, self).__init__()
        switches = []
        self.initial_conv = ConvBnLelu(3, transformation_filters, bn=False)
        self.proc_conv = ConvBnLelu(transformation_filters, transformation_filters, bn=False)
        self.final_conv = ConvBnLelu(transformation_filters, 3, bn=False, lelu=False)
        for filters, growth, sw_reduce, sw_proc, trans_count, kernel, layers in zip(switch_filters, switch_growths, switch_reductions, switch_processing_layers, trans_counts, trans_kernel_sizes, trans_layers):
            multiplx_fn = functools.partial(ConvBasisMultiplexer, transformation_filters, filters, growth, sw_reduce, sw_proc, trans_count)
            switches.append(ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                       pre_transform_block=functools.partial(ConvBnLelu, transformation_filters, transformation_filters, bn=False, bias=False),
                                                       transform_block=functools.partial(MultiConvBlock, transformation_filters, transformation_filters, transformation_filters, kernel_size=kernel, depth=layers),
                                                       transform_count=trans_count, init_temp=initial_temp, enable_negative_transforms=enable_negative_transforms,
                                                       add_scalable_noise_to_transforms=add_scalable_noise_to_transforms, init_scalar=1))
        # Initialize the transforms with a lesser weight, since they are repeatedly added on to the resultant image.
        initialize_weights([s.transforms for s in switches], .2 / len(switches))

        self.switches = nn.ModuleList(switches)
        self.transformation_counts = trans_counts
        self.init_temperature = initial_temp
        self.final_temperature_step = final_temperature_step
        self.heightened_temp_min = heightened_temp_min
        self.heightened_final_step = heightened_final_step
        self.attentions = None
        self.upsample_factor = upsample_factor

    def forward(self, x):
        x = self.initial_conv(x)

        self.attentions = []
        for i, sw in enumerate(self.switches):
            x, att = sw.forward(x, True)
            self.attentions.append(att)

        if self.upsample_factor > 1:
            x = F.interpolate(x, scale_factor=self.upsample_factor, mode="nearest")

        x = self.proc_conv(x)
        x = self.final_conv(x)
        return x,

    def set_temperature(self, temp):
        [sw.set_temperature(temp) for sw in self.switches]

    def update_for_step(self, step, experiments_path='.'):
        if self.attentions:
            temp = max(1, int(self.init_temperature * (self.final_temperature_step - step) / self.final_temperature_step))
            if temp == 1 and self.heightened_final_step and self.heightened_final_step != 1:
                # Once the temperature passes (1) it enters an inverted curve to match the linear curve from above.
                # without this, the attention specificity "spikes" incredibly fast in the last few iterations.
                h_steps_total = self.heightened_final_step - self.final_temperature_step
                h_steps_current = min(step - self.final_temperature_step, h_steps_total)
                # The "gap" will represent the steps that need to be traveled as a linear function.
                h_gap = 1 / self.heightened_temp_min
                temp = h_gap * h_steps_current / h_steps_total
                # Invert temperature to represent reality on this side of the curve
                temp = 1 / temp
            self.set_temperature(temp)
            if step % 50 == 0:
                [save_attention_to_image(experiments_path, self.attentions[i], self.transformation_counts[i], step, "a%i" % (i+1,)) for i in range(len(self.switches))]

    def get_debug_values(self, step):
        temp = self.switches[0].switch.temperature
        mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
        means = [i[0] for i in mean_hists]
        hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
        val = {"switch_temperature": temp}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
        return val


class Interpolate(nn.Module):
    def __init__(self, factor):
        super(Interpolate, self).__init__()
        self.factor = factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor)


class ConfigurableSwitchedResidualGenerator3(nn.Module):
    def __init__(self, trans_counts,
                 trans_kernel_sizes,
                 trans_layers, transformation_filters, initial_temp=20, final_temperature_step=50000,
                 heightened_temp_min=1,
                 heightened_final_step=50000, upsample_factor=1, enable_negative_transforms=False,
                 add_scalable_noise_to_transforms=False):
        super(ConfigurableSwitchedResidualGenerator3, self).__init__()
        switches = []
        for trans_count, kernel, layers in zip(trans_counts, trans_kernel_sizes, trans_layers):
            multiplx_fn = functools.partial(SpineNetMultiplexer, 3)
            switches.append(ConfigurableSwitchComputer(base_filters=3, multiplexer_net=multiplx_fn,
                                                       pre_transform_block=functools.partial(nn.Sequential,
                                                                                             ConvBnLelu(3, transformation_filters, kernel_size=1, stride=4, bn=False, lelu=False, bias=False),
                                                                                             ResidualDenseBlock_5C(
                                                                                                 transformation_filters),
                                                                                             ResidualDenseBlock_5C(
                                                                                                 transformation_filters)),
                                                       transform_block=functools.partial(nn.Sequential,
                                                                                         ResidualDenseBlock_5C(transformation_filters),
                                                                                         Interpolate(4),
                                                                                         ConvBnLelu(transformation_filters, transformation_filters // 2, kernel_size=3, bias=False, bn=False),
                                                                                         ConvBnLelu(transformation_filters // 2, 3, kernel_size=1, bias=False, bn=False, lelu=False)),
                                                       transform_count=trans_count, init_temp=initial_temp,
                                                       enable_negative_transforms=enable_negative_transforms,
                                                       add_scalable_noise_to_transforms=add_scalable_noise_to_transforms,
                                                       init_scalar=.01))

        self.switches = nn.ModuleList(switches)
        self.transformation_counts = trans_counts
        self.init_temperature = initial_temp
        self.final_temperature_step = final_temperature_step
        self.heightened_temp_min = heightened_temp_min
        self.heightened_final_step = heightened_final_step
        self.attentions = None
        self.upsample_factor = upsample_factor

    def forward(self, x):
        if self.upsample_factor > 1:
            x = F.interpolate(x, scale_factor=self.upsample_factor, mode="nearest")

        self.attentions = []
        for i, sw in enumerate(self.switches):
            x, att = sw.forward(x, True)
            self.attentions.append(att)

        return x,

    def set_temperature(self, temp):
        [sw.set_temperature(temp) for sw in self.switches]

    def update_for_step(self, step, experiments_path='.'):
        if self.attentions:
            temp = max(1, int(
                self.init_temperature * (self.final_temperature_step - step) / self.final_temperature_step))
            if temp == 1 and self.heightened_final_step and self.heightened_final_step != 1:
                # Once the temperature passes (1) it enters an inverted curve to match the linear curve from above.
                # without this, the attention specificity "spikes" incredibly fast in the last few iterations.
                h_steps_total = self.heightened_final_step - self.final_temperature_step
                h_steps_current = min(step - self.final_temperature_step, h_steps_total)
                # The "gap" will represent the steps that need to be traveled as a linear function.
                h_gap = 1 / self.heightened_temp_min
                temp = h_gap * h_steps_current / h_steps_total
                # Invert temperature to represent reality on this side of the curve
                temp = 1 / temp
            self.set_temperature(temp)
            if step % 50 == 0:
                [save_attention_to_image(experiments_path, self.attentions[i], self.transformation_counts[i], step,
                                         "a%i" % (i + 1,)) for i in range(len(self.switches))]

    def get_debug_values(self, step):
        temp = self.switches[0].switch.temperature
        mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
        means = [i[0] for i in mean_hists]
        hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
        val = {"switch_temperature": temp}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
        return val