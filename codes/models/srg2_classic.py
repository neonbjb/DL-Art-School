import os

import torch
import torchvision
from matplotlib import cm
from torch import nn
import torch.nn.functional as F
import functools
from collections import OrderedDict

from models.SwitchedResidualGenerator_arch import HalvingProcessingBlock, ConfigurableSwitchComputer
from models.arch_util import ConvBnLelu, ConvGnSilu, ExpansionBlock, MultiConvBlock
from models.switched_conv.switched_conv import BareConvSwitch, AttentionNorm
from utils.util import checkpoint


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
                 add_scalable_noise_to_transforms=False, for_video=False):
        super(ConfigurableSwitchedResidualGenerator2, self).__init__()
        switches = []
        self.for_video = for_video
        if for_video:
            self.initial_conv = ConvBnLelu(6, transformation_filters, stride=upsample_factor, norm=False, activation=False, bias=True)
        else:
            self.initial_conv = ConvBnLelu(3, transformation_filters, norm=False, activation=False, bias=True)
        self.upconv1 = ConvBnLelu(transformation_filters, transformation_filters, norm=False, bias=True)
        self.upconv2 = ConvBnLelu(transformation_filters, transformation_filters, norm=False, bias=True)
        self.hr_conv = ConvBnLelu(transformation_filters, transformation_filters, norm=False, bias=True)
        self.final_conv = ConvBnLelu(transformation_filters, 3, norm=False, activation=False, bias=True)
        for _ in range(switch_depth):
            multiplx_fn = functools.partial(ConvBasisMultiplexer, transformation_filters, switch_filters, switch_reductions, switch_processing_layers, trans_counts)
            pretransform_fn = functools.partial(ConvBnLelu, transformation_filters, transformation_filters, norm=False, bias=False, weight_init_factor=.1)
            transform_fn = functools.partial(MultiConvBlock, transformation_filters, int(transformation_filters * 1.5), transformation_filters, kernel_size=trans_kernel_sizes, depth=trans_layers, weight_init_factor=.1)
            switches.append(ConfigurableSwitchComputer(transformation_filters, multiplx_fn, attention_norm=True,
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

    def forward(self, x, ref=None):
        if self.for_video:
            x_lg = F.interpolate(x, scale_factor=self.upsample_factor, mode="bicubic")
            if ref is None:
                ref = torch.zeros_like(x_lg)
            x_lg = torch.cat([x_lg, ref], dim=1)
        else:
            x_lg = x
        x = self.initial_conv(x_lg)

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
    def __init__(self, factor, mode="nearest"):
        super(Interpolate, self).__init__()
        self.factor = factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode=self.mode)

