import models.archs.SwitchedResidualGenerator_arch as srg
import torch
import torch.nn as nn
from switched_conv_util import save_attention_to_image
from switched_conv import compute_attention_specificity
from models.archs.arch_util import ConvGnLelu, ExpansionBlock, MultiConvBlock
import functools
import torch.nn.functional as F

# Some notes about this new architecture:
# 1) Discriminator is going to need to get update_for_step() called.
# 2) Not sure if pixgan part of discriminator is going to work properly, make sure to test at multiple add levels.
# 3) Also not sure if growth modules will be properly saved/trained, be sure to test this.
# 4) start_step will need to get set properly when constructing these models, even when resuming - OR another method needs to be added to resume properly.

class GrowingSRGBase(nn.Module):
    def __init__(self, progressive_step_schedule, switch_reductions, growth_fade_in_steps, switch_filters, switch_processing_layers, trans_counts,
                 trans_layers, transformation_filters, initial_temp=20, final_temperature_step=50000, upsample_factor=1,
                 add_scalable_noise_to_transforms=False, start_step=0):
        super(GrowingSRGBase, self).__init__()
        switches = []
        self.initial_conv = ConvGnLelu(3, transformation_filters, norm=False, activation=False, bias=True)
        self.upconv1 = ConvGnLelu(transformation_filters, transformation_filters, norm=False, bias=True)
        self.upconv2 = ConvGnLelu(transformation_filters, transformation_filters, norm=False, bias=True)
        self.hr_conv = ConvGnLelu(transformation_filters, transformation_filters, norm=False, bias=True)
        self.final_conv = ConvGnLelu(transformation_filters, 3, norm=False, activation=False, bias=True)

        self.switch_filters = switch_filters
        self.switch_processing_layers = switch_processing_layers
        self.trans_layers = trans_layers
        self.transformation_filters = transformation_filters
        self.progressive_schedule = progressive_step_schedule
        self.switch_reductions = switch_reductions  # This lists the reductions for all switches (even ones not activated yet).
        self.growth_fade_in_per_step = 1 / growth_fade_in_steps
        self.transformation_counts = trans_counts
        self.init_temperature = initial_temp
        self.final_temperature_step = final_temperature_step
        self.attentions = None
        self.upsample_factor = upsample_factor
        self.add_noise_to_transform = add_scalable_noise_to_transforms
        self.start_step = start_step
        self.latest_step = start_step
        self.fades = []
        self.counter = 0
        assert self.upsample_factor == 2 or self.upsample_factor == 4

        switches = []
        for i, (step, reductions) in enumerate(zip(progressive_step_schedule, switch_reductions)):
            multiplx_fn = functools.partial(srg.ConvBasisMultiplexer, self.transformation_filters, self.switch_filters,
                                        reductions, self.switch_processing_layers, self.transformation_counts)
            pretransform_fn = functools.partial(ConvGnLelu, self.transformation_filters, self.transformation_filters, norm=False,
                                                bias=False, weight_init_factor=.1)
            transform_fn = functools.partial(srg.MultiConvBlock, self.transformation_filters, int(self.transformation_filters * 1.5),
                                             self.transformation_filters, kernel_size=3, depth=self.trans_layers,
                                             weight_init_factor=.1)
            switches.append(srg.ConfigurableSwitchComputer(self.transformation_filters, multiplx_fn,
                                                           pre_transform_block=pretransform_fn,
                                                           transform_block=transform_fn,
                                                           transform_count=self.transformation_counts, init_temp=self.init_temperature,
                                                           add_scalable_noise_to_transforms=self.add_noise_to_transform,
                                                           attention_norm=False))
        self.progressive_switches = nn.ModuleList(switches)

    def get_param_groups(self):
        param_groups = []
        base_param_group = []
        for k, v in self.named_parameters():
            if "progressive_switches" not in k and v.requires_grad:
                base_param_group.append(v)
        param_groups.append({'params': base_param_group})
        for i, sw in enumerate(self.progressive_switches):
            sw_param_group = []
            for k, v in sw.named_parameters():
                if v.requires_grad:
                    sw_param_group.append(v)
            param_groups.append({'params': sw_param_group})
        return param_groups

    # This is a hacky way of modifying the underlying model while training. Since changing the model means changing
    # the optimizer and the scheduler, these things are fed in. For ProgressiveSrg, this function adds an additional
# switch to the end of the chain with depth=3 and an online time set at the end fo the function.
    def update_model(self, opt, sched):
        multiplx_fn = functools.partial(srg.ConvBasisMultiplexer, self.transformation_filters, self.switch_filters,
                                    3, self.switch_processing_layers, self.transformation_counts)
        pretransform_fn = functools.partial(ConvGnLelu, self.transformation_filters, self.transformation_filters, norm=False,
                                            bias=False, weight_init_factor=.1)
        transform_fn = functools.partial(srg.MultiConvBlock, self.transformation_filters, int(self.transformation_filters * 1.5),
                                         self.transformation_filters, kernel_size=3, depth=self.trans_layers,
                                         weight_init_factor=.1)
        new_sw = srg.ConfigurableSwitchComputer(self.transformation_filters, multiplx_fn,
                                                           pre_transform_block=pretransform_fn,
                                                           transform_block=transform_fn,
                                                           transform_count=self.transformation_counts, init_temp=self.init_temperature,
                                                           add_scalable_noise_to_transforms=self.add_noise_to_transform,
                                                           attention_norm=False).to('cuda')
        self.progressive_switches.append(new_sw)
        new_sw_param_group = []
        for k, v in new_sw.named_parameters():
            if v.requires_grad:
                new_sw_param_group.append(v)
        opt.add_param_group({'params': new_sw_param_group})
        self.progressive_schedule.append(150000)
        sched.group_starts.append(150000)

    def get_progressive_starts(self):
        # The base param group starts at step 0, the rest are defined via progressive_switches.
        return [0] + self.progressive_schedule

    # This method turns requires_grad on and off for different switches, allowing very large models to be trained while
    # using less memory. When used in conjunction with gradient accumulation, it becomes a form of model parallelism.
    # <groups> controls the proportion of switches that are enabled. 1/groups will be enabled.
    # Switches that are younger than 40000 steps are not eligible to be turned off.
    def do_switched_grad(self, groups=1):
        # If requires_grad is already disabled, don't bother.
        if not self.initial_conv.conv.weight.requires_grad or groups == 1:
            return
        self.counter = (self.counter + 1) % groups
        enabled = []
        for i, sw in enumerate(self.progressive_switches):
            if self.latest_step - self.progressive_schedule[i] > 40000 and i % groups != self.counter:
                for p in sw.parameters():
                    p.requires_grad = False
            else:
                enabled.append(i)
                for p in sw.parameters():
                    p.requires_grad = True

    def forward(self, x):
        self.do_switched_grad(2)

        x = self.initial_conv(x)

        self.attentions = []
        self.fades = []
        self.enabled_switches = 0
        for i, sw in enumerate(self.progressive_switches):
            fade_in = 1 if self.progressive_schedule[i] == 0 else 0
            if self.latest_step > 0 and self.progressive_schedule[i] != 0:
                switch_age = self.latest_step - self.progressive_schedule[i]
                fade_in = min(1, switch_age * self.growth_fade_in_per_step)

            if fade_in > 0:
                self.enabled_switches += 1
                x, att = sw.forward(x, True, fixed_scale=fade_in)
                self.attentions.append(att)
            self.fades.append(fade_in)

        x = self.upconv1(F.interpolate(x, scale_factor=2, mode="nearest"))
        if self.upsample_factor > 2:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.upconv2(x)
        x = self.final_conv(self.hr_conv(x))
        return x, x

    def update_for_step(self, step, experiments_path='.'):
        self.latest_step = step + self.start_step

        # Set the temperature of the switches, per-layer.
        for i, (first_step, sw) in enumerate(zip(self.progressive_schedule, self.progressive_switches)):
            temp_loss_per_step = (self.init_temperature - 1) / self.final_temperature_step
            sw.set_temperature(min(self.init_temperature,
                                   max(self.init_temperature - temp_loss_per_step * (step - first_step), 1)))

        # Save attention images.
        if self.attentions is not None and step % 50 == 0:
            [save_attention_to_image(experiments_path, self.attentions[i], self.transformation_counts, step, "a%i" % (i+1,), l_mult=10) for i in range(len(self.attentions))]

    def get_debug_values(self, step):
        mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
        means = [i[0] for i in mean_hists]
        hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
        val = {}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
            val["switch_%i_temperature" % (i,)] = self.progressive_switches[i].switch.temperature
        for i, f in enumerate(self.fades):
            val["switch_%i_fade" % (i,)] = f
        val["enabled_switches"] = self.enabled_switches
        return val

class DiscriminatorDownsample(nn.Module):
    def __init__(self, base_filters, end_filters):
        self.conv0 = ConvGnLelu(base_filters, end_filters, kernel_size=3, bias=False)
        self.conv1 = ConvGnLelu(end_filters, end_filters, kernel_size=3, stride=2, bias=False)

    def forward(self, x):
        return self.conv1(self.conv0(x))


class DiscriminatorUpsample(nn.Module):
    def __init__(self, base_filters, end_filters):
        self.up = ExpansionBlock(base_filters, end_filters, block=ConvGnLelu)
        self.proc = ConvGnLelu(end_filters, end_filters, bias=False)
        self.collapse = ConvGnLelu(end_filters, 1, bias=True, norm=False, activation=False)

    def forward(self, x, ff):
        x = self.up1(x, ff)
        return x, self.collapse1(self.proc1(x))


class GrowingUnetDiscBase(nn.Module):
    def __init__(self, nf, growing_schedule, growth_fade_in_steps, start_step=0):
        super(GrowingUnetDiscBase, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = ConvGnLelu(3, nf, kernel_size=3, bias=True, activation=False)
        self.conv0_1 = ConvGnLelu(nf, nf, kernel_size=3, stride=2, bias=False)
        # [64, 64, 64]
        self.conv1_0 = ConvGnLelu(nf, nf * 2, kernel_size=3, bias=False)
        self.conv1_1 = ConvGnLelu(nf * 2, nf * 2, kernel_size=3, stride=2, bias=False)

        self.down_base = DiscriminatorDownsample(nf * 2, nf * 4)
        self.up_base = DiscriminatorUpsample(nf * 4, nf * 2)

        self.progressive_schedule = growing_schedule
        self.growth_fade_in_per_step = 1 / growth_fade_in_steps
        self.pnf = nf * 4
        self.downsamples = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])

        for i, step in enumerate(growing_schedule):
            if step >= start_step:
                self.add_layer(i + 1)

    def add_layer(self):
        self.downsamples.append(DiscriminatorDownsample(self.pnf, self.pnf))
        self.upsamples.append(DiscriminatorUpsample(self.pnf, self.pnf))

    def update_for_step(self, step):
        self.latest_step = step

        # Add any new layers as spelled out by the schedule.
        if step != 0:
            for i, s in enumerate(self.progressive_schedule):
                if s == step:
                    self.add_layer(i + 1)

    def forward(self, x, output_feature_vector=False):
        x = self.conv0_0(x)
        x = self.conv0_1(x)
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        base_fea = self.down_base(x)
        x = base_fea

        skips = []
        for down in self.downsamples:
            x = down(x)
            skips.append(x)

        losses = []
        for i, up in enumerate(self.upsamples):
            j = i + 1
            x, loss = up(x, skips[-j])
            losses.append(loss)

        # This variant averages the outputs of the U-net across the upsamples, weighting the contribution
        # to the average less for newly growing levels.
        _, base_loss = self.up_base(x, base_fea)
        res = base_loss.shape[2:]

        mean_weight = 1
        for i, l in enumerate(losses):
            fade_in = 1
            if self.latest_step > 0 and self.progressive_schedule[i] != 0:
                disc_age = self.latest_step - self.progressive_schedule[i]
                fade_in = min(1, disc_age * self.growth_fade_in_per_step)
            mean_weight += fade_in
            base_loss += F.interpolate(l, size=res, mode="bilinear", align_corners=False) * fade_in
        base_loss /= mean_weight

        return base_loss.view(-1, 1)

    def pixgan_parameters(self):
        return 1, 4