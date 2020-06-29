import torch
from torch import nn
from models.archs.SwitchedResidualGenerator_arch import ConvBnLelu, create_sequential_growing_processing_block, MultiConvBlock, initialize_weights
from switched_conv import BareConvSwitch, compute_attention_specificity
from switched_conv_util import save_attention_to_image
from functools import partial
import torch.nn.functional as F


class Switch(nn.Module):
    def __init__(self, transform_block, transform_count, init_temp=20, pass_chain_forward=False, add_scalable_noise_to_transforms=False):
        super(Switch, self).__init__()

        self.transforms = nn.ModuleList([transform_block() for _ in range(transform_count)])
        self.add_noise = add_scalable_noise_to_transforms
        self.pass_chain_forward = pass_chain_forward

        # And the switch itself, including learned scalars
        self.switch = BareConvSwitch(initial_temperature=init_temp)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    # x is the input fed to the transform blocks.
    # m is the output of the multiplexer which will be used to select from those transform blocks.
    # chain is a chain of shared processing outputs used by the individual transforms.
    def forward(self, x, m, chain):
        if self.pass_chain_forward:
            pcf = [t.forward(x, chain) for t in self.transforms]
            xformed = [o[0] for o in pcf]
            atts = [o[1] for o in pcf]
        else:
            if self.add_noise:
                rand_feature = torch.randn_like(x)
                xformed = [t.forward(x, rand_feature) for t in self.transforms]
            else:
                xformed = [t.forward(x) for t in self.transforms]

        # Interpolate the multiplexer across the entire shape of the image.
        m = F.interpolate(m, size=x.shape[2:], mode='nearest')

        outputs, attention = self.switch(xformed, m, True)
        outputs = outputs * self.scale + self.bias

        if self.pass_chain_forward:
            # Apply attention weights to collected [atts] and return the aggregate.
            atts = torch.stack(atts, dim=3)
            attention = atts * attention.unsqueeze(dim=-1)
            attention = torch.flatten(attention, 3)

        return outputs, attention

    def set_temperature(self, temp):
        self.switch.set_attention_temperature(temp)
        if self.pass_chain_forward:
            [t.set_temperature(temp) for t in self.transforms]


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.lelu1 = nn.LeakyReLU(negative_slope=.1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.lelu2 = nn.LeakyReLU(negative_slope=.1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(self.lelu1(self.bn1(x)))
        return self.conv2(self.lelu2(self.bn2(x)))


# Convolutional image processing block that optionally reduces image size by a factor of 2 using stride and performs a
# series of residual-block-like processing operations on it.
class Processor(nn.Module):
    def __init__(self, base_filters, processing_depth, reduce=False):
        super(Processor, self).__init__()
        self.output_filter_count = base_filters * 2 if reduce else base_filters
        self.initial = ConvBnLelu(base_filters, self.output_filter_count, kernel_size=1, stride=2 if reduce else 1)
        self.res_blocks = nn.ModuleList([ResidualBlock(self.output_filter_count) for _ in range(processing_depth)])

    def forward(self, x):
        x = self.initial(x)
        for b in self.res_blocks:
            x = b(x) + x
        return x


# Convolutional image processing block that constricts an input image with a large number of filters to a small number
# of filters over a fixed number of layers.
class Constrictor(nn.Module):
    def __init__(self, filters, output_filters, use_bn=False):
        super(Constrictor, self).__init__()
        assert(filters > output_filters)
        gap = filters - output_filters
        gap_div_4 = int(gap / 4)
        self.cbl1 = ConvBnLelu(filters, filters - (gap_div_4 * 2), bn=use_bn)
        self.cbl2 = ConvBnLelu(filters - (gap_div_4 * 2), filters - (gap_div_4 * 3), bn=use_bn)
        self.cbl3 = ConvBnLelu(filters - (gap_div_4 * 3), output_filters)

    def forward(self, x):
        x = self.cbl1(x)
        x = self.cbl2(x)
        x = self.cbl3(x)
        return x


class RecursiveSwitchedTransform(nn.Module):
    def __init__(self, transform_filters, filters_count_list, nesting_depth, transforms_at_leaf,
                 trans_kernel_size, trans_num_layers, trans_scale_init=1, initial_temp=20, add_scalable_noise_to_transforms=False):
        super(RecursiveSwitchedTransform, self).__init__()

        self.depth = nesting_depth
        at_leaf = (self.depth == 0)
        if at_leaf:
            transform = partial(MultiConvBlock, transform_filters, transform_filters, transform_filters, kernel_size=trans_kernel_size, depth=trans_num_layers, scale_init=trans_scale_init)
        else:
            transform = partial(RecursiveSwitchedTransform, transform_filters, filters_count_list,
                                nesting_depth - 1, transforms_at_leaf, trans_kernel_size, trans_num_layers, trans_scale_init, initial_temp, add_scalable_noise_to_transforms)
        selection_breadth = transforms_at_leaf if at_leaf else 2
        self.switch = Switch(transform, selection_breadth, initial_temp, pass_chain_forward=not at_leaf, add_scalable_noise_to_transforms=add_scalable_noise_to_transforms)
        self.multiplexer = Constrictor(filters_count_list[self.depth], selection_breadth)

    def forward(self, x, processing_trunk_chain):
        proc_out = processing_trunk_chain[self.depth]
        m = self.multiplexer(proc_out)
        return self.switch(x, m, processing_trunk_chain)

    def set_temperature(self, temp):
        self.switch.set_temperature(temp)


class NestedSwitchComputer(nn.Module):
    def __init__(self, transform_filters, switch_base_filters, num_switch_processing_layers, nesting_depth, transforms_at_leaf,
                 trans_kernel_size, trans_num_layers, trans_scale_init, initial_temp=20, add_scalable_noise_to_transforms=False):
        super(NestedSwitchComputer, self).__init__()

        processing_trunk = []
        filters = []
        current_filters = switch_base_filters
        for _ in range(nesting_depth):
            processing_trunk.append(Processor(current_filters, num_switch_processing_layers, reduce=True))
            current_filters = processing_trunk[-1].output_filter_count
            filters.append(current_filters)

        self.multiplexer_init_conv = nn.Conv2d(transform_filters, switch_base_filters, kernel_size=7, padding=3)
        self.processing_trunk = nn.ModuleList(processing_trunk)
        self.switch = RecursiveSwitchedTransform(transform_filters, filters, nesting_depth-1, transforms_at_leaf, trans_kernel_size, trans_num_layers-1, trans_scale_init, initial_temp=initial_temp, add_scalable_noise_to_transforms=add_scalable_noise_to_transforms)
        self.anneal = ConvBnLelu(transform_filters, transform_filters, kernel_size=1, bn=False)

    def forward(self, x):
        trunk = []
        trunk_input = self.multiplexer_init_conv(x)
        for m in self.processing_trunk:
            trunk_input = m.forward(trunk_input)
            trunk.append(trunk_input)

        x, att = self.switch.forward(x, trunk)
        return self.anneal(x), att

    def set_temperature(self, temp):
        self.switch.set_temperature(temp)


class NestedSwitchedGenerator(nn.Module):
    def __init__(self, switch_filters, switch_reductions, switch_processing_layers, trans_counts, trans_kernel_sizes,
                 trans_layers, transformation_filters, initial_temp=20, final_temperature_step=50000, heightened_temp_min=1,
                 heightened_final_step=50000, upsample_factor=1, add_scalable_noise_to_transforms=False):
        super(NestedSwitchedGenerator, self).__init__()
        self.initial_conv = ConvBnLelu(3, transformation_filters, bn=False)
        self.final_conv = ConvBnLelu(transformation_filters, 3, bn=False)

        switches = []
        for sw_reduce, sw_proc, trans_count, kernel, layers in zip(switch_reductions, switch_processing_layers, trans_counts, trans_kernel_sizes, trans_layers):
            switches.append(NestedSwitchComputer(transform_filters=transformation_filters, switch_base_filters=switch_filters, num_switch_processing_layers=sw_proc,
                                                 nesting_depth=sw_reduce, transforms_at_leaf=trans_count, trans_kernel_size=kernel, trans_num_layers=layers,
                                                 trans_scale_init=.2/len(switch_reductions), initial_temp=initial_temp, add_scalable_noise_to_transforms=add_scalable_noise_to_transforms))
        initialize_weights(switches, 1)
        self.switches = nn.ModuleList(switches)

        self.transformation_counts = trans_counts
        self.init_temperature = initial_temp
        self.final_temperature_step = final_temperature_step
        self.heightened_temp_min = heightened_temp_min
        self.heightened_final_step = heightened_final_step
        self.attentions = None
        self.upsample_factor = upsample_factor

    def forward(self, x):
        # This network is entirely a "repair" network and operates on full-resolution images. Upsample first if that
        # is called for, then repair.
        if self.upsample_factor > 1:
            x = F.interpolate(x, scale_factor=self.upsample_factor, mode="nearest")

        x = self.initial_conv(x)

        self.attentions = []
        for i, sw in enumerate(self.switches):
            sw_out, att = sw.forward(x)
            self.attentions.append(att)
            x = x + sw_out

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
        temp = self.switches[0].switch.switch.switch.temperature
        mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
        means = [i[0] for i in mean_hists]
        hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
        val = {"switch_temperature": temp}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
        return val