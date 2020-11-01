import os

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import functools
from collections import OrderedDict

from torch.nn import init

from models.archs.arch_util import ConvBnLelu, ConvGnSilu
from utils.util import checkpoint


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class AttentionNorm(nn.Module):
    def __init__(self, group_size, accumulator_size=128):
        super(AttentionNorm, self).__init__()
        self.accumulator_desired_size = accumulator_size
        self.group_size = group_size
        # These are all tensors so that they get saved with the graph.
        self.accumulator = nn.Parameter(torch.zeros(accumulator_size, group_size), requires_grad=False)
        self.accumulator_index = nn.Parameter(torch.zeros(1, dtype=torch.long, device='cpu'), requires_grad=False)
        self.accumulator_filled = nn.Parameter(torch.zeros(1, dtype=torch.bool, device='cpu'), requires_grad=False)

    # Returns tensor of shape (group,) with a normalized mean across the accumulator in the range [0,1]. The intent
    # is to divide your inputs by this value.
    def compute_buffer_norm(self):
        if self.accumulator_filled:
            return torch.mean(self.accumulator, dim=0)
        else:
            return torch.ones(self.group_size, device=self.accumulator.device)

    def add_norm_to_buffer(self, x):
        flat = x.sum(dim=[0, 1, 2], keepdim=True)
        norm = flat / torch.mean(flat)

        # This often gets reset in GAN mode. We *never* want gradient accumulation in this parameter.
        self.accumulator.requires_grad = False
        self.accumulator[self.accumulator_index] = norm.detach()
        self.accumulator_index += 1
        if self.accumulator_index >= self.accumulator_desired_size:
            self.accumulator_index *= 0
            self.accumulator_filled |= True

    # Input into forward is an attention tensor of shape (batch,width,height,groups)
    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 4
        # Push the accumulator to the right device on the first iteration.
        if self.accumulator.device != x.device:
            self.accumulator = self.accumulator.to(x.device)

        self.add_norm_to_buffer(x)
        norm = self.compute_buffer_norm()
        x = x / norm

        # Need to re-normalize x so that the groups dimension sum to 1, just like when it was fed in.
        groups_sum = x.sum(dim=3, keepdim=True)
        return x / groups_sum


class BareConvSwitch(nn.Module):
    """
    Initializes the ConvSwitch.
      initial_temperature: The initial softmax temperature of the attention mechanism. For training from scratch, this
                           should be set to a high number, for example 30.
      attention_norm:      If specified, the AttentionNorm layer applied immediately after Softmax.
    """

    def __init__(
        self,
        initial_temperature=1,
        attention_norm=None
    ):
        super(BareConvSwitch, self).__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.temperature = initial_temperature
        self.attention_norm = attention_norm

        initialize_weights(self)

    def set_attention_temperature(self, temp):
        self.temperature = temp

    # SwitchedConv.forward takes these arguments;
    # conv_group:      List of inputs (len=n) to the switch, each with shape (b,f,w,h)
    # conv_attention:  Attention computation as an output from a conv layer, of shape (b,n,w,h). Before softmax
    # output_attention_weights: If True, post-softmax attention weights are returned.
    def forward(self, conv_group, conv_attention, output_attention_weights=False):
        # Stack up the conv_group input first and permute it to (batch, width, height, filter, groups)
        conv_outputs = torch.stack(conv_group, dim=0).permute(1, 3, 4, 2, 0)

        conv_attention = conv_attention.permute(0, 2, 3, 1)
        conv_attention = self.softmax(conv_attention / self.temperature)
        if self.attention_norm:
            conv_attention = self.attention_norm(conv_attention)

        # conv_outputs shape:   (batch, width, height, filters, groups)
        # conv_attention shape: (batch, width, height, groups)
        # We want to format them so that we can matmul them together to produce:
        # desired shape:        (batch, width, height, filters)
        # Note: conv_attention will generally be cast to float32 regardless of the input type, so cast conv_outputs to
        #       float32 as well to match it.
        if self.training:
            # Doing it all in one op is substantially faster - better for training.
            attention_result = torch.einsum(
                "...ij,...j->...i", [conv_outputs.float(), conv_attention]
            )
        else:
            # eval_mode substantially reduces the GPU memory required to compute the attention result by performing the
            # attention multiplications one at a time. This is probably necessary for large images and attention breadths.
            attention_result = conv_outputs[:, :, :, :, 0] * conv_attention[:, :, :, 0].unsqueeze(dim=-1)
            for i in range(1, conv_attention.shape[-1]):
                attention_result += conv_outputs[:, :, :, :, i] * conv_attention[:, :, :, i].unsqueeze(dim=-1)

        # Remember to shift the filters back into the expected slot.
        if output_attention_weights:
            return attention_result.permute(0, 3, 1, 2), conv_attention
        else:
            return attention_result.permute(0, 3, 1, 2)


class MultiConvBlock(nn.Module):
    def __init__(self, filters_in, filters_mid, filters_out, kernel_size, depth, scale_init=1.0, norm=False, weight_init_factor=1):
        assert depth >= 2
        super(MultiConvBlock, self).__init__()
        self.noise_scale = nn.Parameter(torch.full((1,), fill_value=.01))
        self.bnconvs = nn.ModuleList([ConvBnLelu(filters_in, filters_mid, kernel_size, norm=norm, bias=False, weight_init_factor=weight_init_factor)] +
                                     [ConvBnLelu(filters_mid, filters_mid, kernel_size, norm=norm, bias=False, weight_init_factor=weight_init_factor) for i in range(depth - 2)] +
                                     [ConvBnLelu(filters_mid, filters_out, kernel_size, activation=False, norm=False, bias=False, weight_init_factor=weight_init_factor)])
        self.scale = nn.Parameter(torch.full((1,), fill_value=scale_init))
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=False)

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
        self.bnconv1 = ConvGnSilu(filters, filters * 2, stride=2, norm=False, bias=False)
        self.bnconv2 = ConvGnSilu(filters * 2, filters * 2, norm=True, bias=False)

    def forward(self, x):
        x = self.bnconv1(x)
        return self.bnconv2(x)


# Block that upsamples 2x and reduces incoming filters by 2x. It preserves structure by taking a passthrough feed
# along with the feature representation.
class ExpansionBlock(nn.Module):
    def __init__(self, filters_in, filters_out=None, block=ConvGnSilu):
        super(ExpansionBlock, self).__init__()
        if filters_out is None:
            filters_out = filters_in // 2
        self.decimate = block(filters_in, filters_out, kernel_size=1, bias=False, activation=False, norm=True)
        self.process_passthrough = block(filters_out, filters_out, kernel_size=3, bias=True, activation=False, norm=True)
        self.conjoin = block(filters_out*2, filters_out, kernel_size=3, bias=False, activation=True, norm=False)
        self.process = block(filters_out, filters_out, kernel_size=3, bias=False, activation=True, norm=True)

    # input is the feature signal with shape  (b, f, w, h)
    # passthrough is the structure signal with shape (b, f/2, w*2, h*2)
    # output is conjoined upsample with shape (b, f/2, w*2, h*2)
    def forward(self, input, passthrough):
        x = F.interpolate(input, scale_factor=2, mode="nearest")
        x = self.decimate(x)
        p = self.process_passthrough(passthrough)
        x = self.conjoin(torch.cat([x, p], dim=1))
        return self.process(x)


# This is a classic u-net architecture with the goal of assigning each individual pixel an individual transform
# switching set.
class ConvBasisMultiplexer(nn.Module):
    def __init__(self, input_channels, base_filters, reductions, processing_depth, multiplexer_channels, use_gn=True):
        super(ConvBasisMultiplexer, self).__init__()
        self.filter_conv = ConvGnSilu(input_channels, base_filters, bias=True)
        self.reduction_blocks = nn.ModuleList([HalvingProcessingBlock(base_filters * 2 ** i) for i in range(reductions)])
        reduction_filters = base_filters * 2 ** reductions
        self.processing_blocks = nn.Sequential(OrderedDict([('block%i' % (i,), ConvGnSilu(reduction_filters, reduction_filters, bias=False)) for i in range(processing_depth)]))
        self.expansion_blocks = nn.ModuleList([ExpansionBlock(reduction_filters // (2 ** i)) for i in range(reductions)])

        gap = base_filters - multiplexer_channels
        cbl1_out = ((base_filters - (gap // 2)) // 4) * 4   # Must be multiples of 4 to use with group norm.
        self.cbl1 = ConvGnSilu(base_filters, cbl1_out, norm=use_gn, bias=False, num_groups=4)
        cbl2_out = ((base_filters - (3 * gap // 4)) // 4) * 4
        self.cbl2 = ConvGnSilu(cbl1_out, cbl2_out, norm=use_gn, bias=False, num_groups=4)
        self.cbl3 = ConvGnSilu(cbl2_out, multiplexer_channels, bias=True, norm=False)

    def forward(self, x):
        x = self.filter_conv(x)
        reduction_identities = []
        for b in self.reduction_blocks:
            reduction_identities.append(x)
            x = b(x)
        x = self.processing_blocks(x)
        for i, b in enumerate(self.expansion_blocks):
            x = b(x, reduction_identities[-i - 1])

        x = self.cbl1(x)
        x = self.cbl2(x)
        x = self.cbl3(x)
        return x


class ConfigurableSwitchComputer(nn.Module):
    def __init__(self, base_filters, multiplexer_net, pre_transform_block, transform_block, transform_count, init_temp=20,
                 add_scalable_noise_to_transforms=False):
        super(ConfigurableSwitchComputer, self).__init__()

        tc = transform_count
        self.multiplexer = multiplexer_net(tc)

        self.pre_transform = pre_transform_block()
        self.transforms = nn.ModuleList([transform_block() for _ in range(transform_count)])
        self.add_noise = add_scalable_noise_to_transforms
        self.noise_scale = nn.Parameter(torch.full((1,), float(1e-3)))

        # And the switch itself, including learned scalars
        self.switch = BareConvSwitch(initial_temperature=init_temp, attention_norm=AttentionNorm(transform_count, accumulator_size=16 * transform_count))
        self.switch_scale = nn.Parameter(torch.full((1,), float(1)))
        self.post_switch_conv = ConvBnLelu(base_filters, base_filters, norm=False, bias=True)
        # The post_switch_conv gets a low scale initially. The network can decide to magnify it (or not)
        # depending on its needs.
        self.psc_scale = nn.Parameter(torch.full((1,), float(.1)))

    def forward(self, x, output_attention_weights=True):
        identity = x
        if self.add_noise:
            rand_feature = torch.randn_like(x) * self.noise_scale
            x = x + rand_feature

        x = self.pre_transform(x)
        xformed = [t.forward(x) for t in self.transforms]

        m = self.multiplexer(identity)

        outputs, attention = self.switch(xformed, m, True)
        outputs = identity + outputs * self.switch_scale
        outputs = outputs + self.post_switch_conv(outputs) * self.psc_scale
        if output_attention_weights:
            return outputs, attention
        else:
            return outputs

    def set_temperature(self, temp):
        self.switch.set_attention_temperature(temp)


def compute_attention_specificity(att_weights, topk=3):
    att = att_weights.detach()
    vals, indices = torch.topk(att, topk, dim=-1)
    avg = torch.sum(vals, dim=-1)
    avg = avg.flatten().mean()
    return avg.item(), indices.flatten().detach()


# Copied from torchvision.utils.save_image. Allows specifying pixel format.
def save_image(tensor, fp, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None, pix_format=None):
    from PIL import Image
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                                       normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr, mode=pix_format).convert('RGB')
    im.save(fp, format=format)


def save_attention_to_image(folder, attention_out, attention_size, step, fname_part="map", l_mult=1.0):
    magnitude, indices = torch.topk(attention_out, 1, dim=-1)
    magnitude = magnitude.squeeze(3)
    indices = indices.squeeze(3)
    # indices is an integer tensor (b,w,h) where values are on the range [0,attention_size]
    # magnitude is a float tensor (b,w,h) [0,1] representing the magnitude of that attention.
    # Use HSV colorspace to show this. Hue is mapped to the indices, Lightness is mapped to intensity,
    # Saturation is left fixed.
    hue = indices.float() / attention_size
    saturation = torch.full_like(hue, .8)
    value = magnitude * l_mult
    hsv_img = torch.stack([hue, saturation, value], dim=1)

    output_path=os.path.join(folder, "attention_maps", fname_part)
    os.makedirs(output_path, exist_ok=True)
    save_image(hsv_img, os.path.join(output_path, "attention_map_%i.png" % (step,)), pix_format="HSV")


def save_attention_to_image_rgb(output_folder, attention_out, attention_size, file_prefix, step, cmap_discrete_name='viridis'):
    magnitude, indices = torch.topk(attention_out, 3, dim=-1)
    magnitude = magnitude.cpu()
    indices = indices.cpu()
    magnitude /= torch.max(torch.abs(torch.min(magnitude)), torch.abs(torch.max(magnitude)))
    colormap = cm.get_cmap(cmap_discrete_name, attention_size)
    colormap_mag = cm.get_cmap(cmap_discrete_name)
    os.makedirs(os.path.join(output_folder), exist_ok=True)
    for i in range(3):
        img = torch.tensor(colormap(indices[:,:,:,i].detach().numpy()))
        img = img.permute((0, 3, 1, 2))
        save_image(img, os.path.join(output_folder, file_prefix + "_%i_%s.png" % (step, "rgb_%i" % (i,))), pix_format="RGBA")

        mag_image = torch.tensor(colormap_mag(magnitude[:,:,:,i].detach().numpy()))
        mag_image = mag_image.permute((0, 3, 1, 2))
        save_image(mag_image, os.path.join(output_folder, file_prefix + "_%i_%s.png" % (step, "mag_%i" % (i,))), pix_format="RGBA")


class ConfigurableSwitchedResidualGenerator2(nn.Module):
    def __init__(self, switch_depth, switch_filters, switch_reductions, switch_processing_layers, trans_counts, trans_kernel_sizes,
                 trans_layers, transformation_filters, initial_temp=20, final_temperature_step=50000, heightened_temp_min=1,
                 heightened_final_step=50000, upsample_factor=1,
                 add_scalable_noise_to_transforms=False):
        super(ConfigurableSwitchedResidualGenerator2, self).__init__()
        switches = []
        self.initial_conv = ConvBnLelu(3, transformation_filters, norm=False, activation=False, bias=True)
        self.upconv1 = ConvBnLelu(transformation_filters, transformation_filters, norm=False, bias=True)
        self.upconv2 = ConvBnLelu(transformation_filters, transformation_filters, norm=False, bias=True)
        self.hr_conv = ConvBnLelu(transformation_filters, transformation_filters, norm=False, bias=True)
        self.final_conv = ConvBnLelu(transformation_filters, 3, norm=False, activation=False, bias=True)
        for _ in range(switch_depth):
            multiplx_fn = functools.partial(ConvBasisMultiplexer, transformation_filters, switch_filters, switch_reductions, switch_processing_layers, trans_counts)
            pretransform_fn = functools.partial(ConvBnLelu, transformation_filters, transformation_filters, norm=False, bias=False, weight_init_factor=.1)
            transform_fn = functools.partial(MultiConvBlock, transformation_filters, int(transformation_filters * 1.5), transformation_filters, kernel_size=trans_kernel_sizes, depth=trans_layers, weight_init_factor=.1)
            switches.append(ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                       pre_transform_block=pretransform_fn, transform_block=transform_fn,
                                                       transform_count=trans_counts, init_temp=initial_temp,
                                                       add_scalable_noise_to_transforms=add_scalable_noise_to_transforms))

        self.switches = nn.ModuleList(switches)
        self.transformation_counts = trans_counts
        self.init_temperature = initial_temp
        self.final_temperature_step = final_temperature_step
        self.heightened_temp_min = heightened_temp_min
        self.heightened_final_step = heightened_final_step
        self.attentions = None
        self.upsample_factor = upsample_factor
        assert self.upsample_factor == 2 or self.upsample_factor == 4

    def forward(self, x):
        x = self.initial_conv(x)

        self.attentions = []
        for i, sw in enumerate(self.switches):
            x, att = checkpoint(sw, x)
            self.attentions.append(att)

        x = self.upconv1(F.interpolate(x, scale_factor=2, mode="nearest"))
        if self.upsample_factor > 2:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.upconv2(x)
        x = self.final_conv(self.hr_conv(x))
        return x

    def set_temperature(self, temp):
        [sw.set_temperature(temp) for sw in self.switches]

    def update_for_step(self, step, experiments_path='.'):
        if self.attentions:
            temp = max(1,
                1 + self.init_temperature * (self.final_temperature_step - step) / self.final_temperature_step)
            if temp == 1 and self.heightened_final_step and step > self.final_temperature_step and \
                    self.heightened_final_step != 1:
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
                [save_attention_to_image(experiments_path, self.attentions[i], self.transformation_counts, step, "a%i" % (i+1,), l_mult=10) for i in  range(len(self.attentions))]

    def get_debug_values(self, step, net_name):
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

