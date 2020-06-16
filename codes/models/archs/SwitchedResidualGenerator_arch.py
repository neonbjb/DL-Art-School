import torch
from torch import nn
from switched_conv import BareConvSwitch, compute_attention_specificity
import torch.nn.functional as F
import functools
from models.archs.arch_util import initialize_weights
from switched_conv_util import save_attention_to_image


class ConvBnLelu(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3, stride=1, lelu=True):
        super(ConvBnLelu, self).__init__()
        padding_map = {1: 0, 3: 1, 5: 2, 7: 3}
        assert kernel_size in padding_map.keys()
        self.conv = nn.Conv2d(filters_in, filters_out, kernel_size, stride, padding_map[kernel_size])
        self.bn = nn.BatchNorm2d(filters_out)
        if lelu:
            self.lelu = nn.LeakyReLU(negative_slope=.1)
        else:
            self.lelu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.lelu:
            return self.lelu(x)
        else:
            return x


class ResidualBranch(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, depth):
        super(ResidualBranch, self).__init__()
        self.bnconvs = nn.ModuleList([ConvBnLelu(filters_in, filters_out, kernel_size)] +
                                     [ConvBnLelu(filters_out, filters_out, kernel_size) for i in range(depth-2)] +
                                     [ConvBnLelu(filters_out, filters_out, kernel_size, lelu=False)])
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        for m in self.bnconvs:
            x = m.forward(x)
        return x * self.scale + self.bias


# VGG-style layer with Conv(stride2)->BN->Activation->Conv->BN->Activation
# Doubles the input filter count.
class HalvingProcessingBlock(nn.Module):
    def __init__(self, filters):
        super(HalvingProcessingBlock, self).__init__()
        self.bnconv1 = ConvBnLelu(filters, filters * 2, stride=2)
        self.bnconv2 = ConvBnLelu(filters * 2, filters * 2)

    def forward(self, x):
        x = self.bnconv1(x)
        return self.bnconv2(x)


class SwitchComputer(nn.Module):
    def __init__(self, channels_in, filters, transform_block, transform_count, reduction_blocks, processing_blocks=0, init_temp=20):
        super(SwitchComputer, self).__init__()
        self.filter_conv = ConvBnLelu(channels_in, filters)
        self.reduction_blocks = nn.ModuleList([HalvingProcessingBlock(filters * 2 ** i) for i in range(reduction_blocks)])
        final_filters = filters * 2 ** reduction_blocks
        self.processing_blocks = nn.ModuleList([ConvBnLelu(final_filters, final_filters) for i in range(processing_blocks)])
        proc_block_filters = max(final_filters // 2, transform_count)
        self.proc_switch_conv = ConvBnLelu(final_filters, proc_block_filters)
        self.final_switch_conv = nn.Conv2d(proc_block_filters, transform_count, 1, 1, 0)

        # Always include the identity transform (all zeros), hence transform_count-10
        self.transforms = nn.ModuleList([transform_block() for i in range(transform_count-1)])

        # And the switch itself
        self.switch = BareConvSwitch(initial_temperature=init_temp)

    def forward(self, x, output_attention_weights=False):
        xformed = [t.forward(x) for t in self.transforms]
        # Append the identity transform.
        xformed.append(torch.zeros_like(xformed[0]))

        multiplexer = self.filter_conv(x)
        for block in self.reduction_blocks:
            multiplexer = block.forward(multiplexer)
        for block in self.processing_blocks:
            multiplexer = block.forward(multiplexer)
        multiplexer = self.proc_switch_conv(multiplexer)
        multiplexer = self.final_switch_conv.forward(multiplexer)
        # Interpolate the multiplexer across the entire shape of the image.
        multiplexer = F.interpolate(multiplexer, size=x.shape[2:], mode='nearest')

        return self.switch(xformed, multiplexer, output_attention_weights)

    def set_temperature(self, temp):
        self.switch.set_attention_temperature(temp)


class ConfigurableSwitchedResidualGenerator(nn.Module):
    def __init__(self, switch_filters, switch_reductions, switch_processing_layers, trans_counts, trans_kernel_sizes, trans_layers, initial_temp=20, final_temperature_step=50000):
        super(ConfigurableSwitchedResidualGenerator, self).__init__()
        switches = []
        for filters, sw_reduce, sw_proc, trans_count, kernel, layers in zip(switch_filters, switch_reductions, switch_processing_layers, trans_counts, trans_kernel_sizes, trans_layers):
            switches.append(SwitchComputer(3, filters, functools.partial(ResidualBranch, 3, 3, kernel_size=kernel, depth=layers), trans_count, sw_reduce, sw_proc, initial_temp))
        initialize_weights(switches, 1)
        # Initialize the transforms with a lesser weight, since they are repeatedly added on to the resultant image.
        initialize_weights([s.transforms for s in switches], .2 / len(switches))
        self.switches = nn.ModuleList(switches)
        self.transformation_counts = trans_counts
        self.init_temperature = initial_temp
        self.final_temperature_step = final_temperature_step

    def forward(self, x):
        self.attentions = []
        for i, sw in enumerate(self.switches):
            sw_out, att = sw.forward(x, True)
            x = x + sw_out
            self.attentions.append(att)
        return x,

    def set_temperature(self, temp):
        [sw.set_temperature(temp) for sw in self.switches]

    def get_debug_values(self, step):
        # Take the chance to update the temperature here.
        temp = max(1, int(self.init_temperature * (self.final_temperature_step - step) / self.final_temperature_step))
        self.set_temperature(temp)

        if step % 250 == 0:
            [save_attention_to_image(self.attentions[i], self.transformation_counts[i], step, "a%i" % (i+1,), l_mult=float(self.transformation_counts[i]/4)) for i in range(len(self.switches))]

        mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
        means = [i[0] for i in mean_hists]
        hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
        val = {"switch_temperature": temp}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
        return val