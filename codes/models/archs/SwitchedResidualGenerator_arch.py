import torch
from torch import nn
from switched_conv import BareConvSwitch, compute_attention_specificity
import torch.nn.functional as F
import functools
from models.archs.arch_util import initialize_weights
import torchvision
from torchvision import transforms


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


# VGG-style layer with Conv->BN->Activation->Conv(stride2)->BN->Activation
class HalvingProcessingBlock(nn.Module):
    def __init__(self, filters):
        super(HalvingProcessingBlock, self).__init__()
        self.bnconv1 = ConvBnLelu(filters, filters)
        self.bnconv2 = ConvBnLelu(filters, filters * 2, stride=2)

    def forward(self, x):
        x = self.bnconv1(x)
        return self.bnconv2(x)


class SwitchComputer(nn.Module):
    def __init__(self, channels_in, filters, transform_block, transform_count, reductions, init_temp=20):
        super(SwitchComputer, self).__init__()
        self.filter_conv = ConvBnLelu(channels_in, filters)
        self.blocks = nn.ModuleList([HalvingProcessingBlock(filters * 2 ** i) for i in range(reductions)])
        final_filters = filters * 2 ** reductions
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
        for block in self.blocks:
            multiplexer = block.forward(multiplexer)
        multiplexer = self.proc_switch_conv(multiplexer)
        multiplexer = self.final_switch_conv.forward(multiplexer)
        # Interpolate the multiplexer across the entire shape of the image.
        multiplexer = F.interpolate(multiplexer, size=x.shape[2:], mode='nearest')

        return self.switch(xformed, multiplexer, output_attention_weights)

    def set_temperature(self, temp):
        self.switch.set_attention_temperature(temp)


class SwitchedResidualGenerator(nn.Module):
    def __init__(self, switch_filters, initial_temp=20, final_temperature_step=50000):
        super(SwitchedResidualGenerator, self).__init__()
        self.switch1 = SwitchComputer(3, switch_filters, functools.partial(ResidualBranch, 3, 3, kernel_size=7, depth=3), 4, 4, initial_temp)
        self.switch2 = SwitchComputer(3, switch_filters, functools.partial(ResidualBranch, 3, 3, kernel_size=5, depth=3), 8, 3, initial_temp)
        self.switch3 = SwitchComputer(3, switch_filters, functools.partial(ResidualBranch, 3, 3, kernel_size=3, depth=3), 16, 2, initial_temp)
        self.switch4 = SwitchComputer(3, switch_filters, functools.partial(ResidualBranch, 3, 3, kernel_size=3, depth=2), 32, 1, initial_temp)
        initialize_weights([self.switch1, self.switch2, self.switch3, self.switch4], 1)
        # Initialize the transforms with a lesser weight, since they are repeatedly added on to the resultant image.
        initialize_weights([self.switch1.transforms, self.switch2.transforms, self.switch3.transforms, self.switch4.transforms], .05)

        self.init_temperature = initial_temp
        self.final_temperature_step = final_temperature_step
        self.running_sum = [0, 0, 0, 0]
        self.running_count = 0

    def forward(self, x):
        sw1, self.a1 = self.switch1.forward(x, True)
        x = x + sw1
        sw2, self.a2 = self.switch2.forward(x, True)
        x = x + sw2
        sw3, self.a3 = self.switch3.forward(x, True)
        x = x + sw3
        sw4, self.a4 = self.switch4.forward(x, True)
        x = x + sw4

        a1mean, _ = compute_attention_specificity(self.a1, 2)
        a2mean, _ = compute_attention_specificity(self.a2, 2)
        a3mean, _ = compute_attention_specificity(self.a3, 2)
        a4mean, _ = compute_attention_specificity(self.a4, 2)
        running_sum = [
            self.running_sum[0] + a1mean,
            self.running_sum[1] + a2mean,
            self.running_sum[2] + a3mean,
            self.running_sum[3] + a4mean,
        ]
        self.running_count += 1

        return (x,)

    def set_temperature(self, temp):
        self.switch1.set_temperature(temp)
        self.switch2.set_temperature(temp)
        self.switch3.set_temperature(temp)
        self.switch4.set_temperature(temp)

    # Copied from torchvision.utils.save_image. Allows specifying pixel format.
    def save_image(self, tensor, fp, nrow=8, padding=2,
                   normalize=False, range=None, scale_each=False, pad_value=0, format=None, pix_format=None):
        from PIL import Image
        grid = torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                         normalize=normalize, range=range, scale_each=scale_each)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr, mode=pix_format).convert('RGB')
        im.save(fp, format=format)

    def convert_attention_indices_to_image(self, attention_out, attention_size, step, fname_part="map", l_mult=1.0):
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

        import os
        os.makedirs("attention_maps/%s" % (fname_part,), exist_ok=True)
        self.save_image(hsv_img, "attention_maps/%s/attention_map_%i.png" % (fname_part, step,), pix_format="HSV")

    def get_debug_values(self, step):
        # Take the chance to update the temperature here.
        temp = max(1, int(self.init_temperature * (self.final_temperature_step - step) / self.final_temperature_step))
        self.set_temperature(temp)

        if step % 250 == 0:
            self.convert_attention_indices_to_image(self.a1, 4, step, "a1")
            self.convert_attention_indices_to_image(self.a2, 8, step, "a2")
            self.convert_attention_indices_to_image(self.a3, 16, step, "a3", 2)
            self.convert_attention_indices_to_image(self.a4, 32, step, "a4", 4)

        val = {"switch_temperature": temp}
        for i in range(len(self.running_sum)):
            val["switch_%i_specificity" % (i,)] = self.running_sum[i] / self.running_count
            self.running_sum[i] = 0
        self.running_count = 0
        return val