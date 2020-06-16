import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
from models.archs.arch_util import PixelUnshuffle
import torchvision
import switched_conv as switched_conv


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True, late_stage_kernel_size=3, late_stage_padding=1):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, late_stage_kernel_size, 1, late_stage_padding, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, late_stage_kernel_size, 1, late_stage_padding, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, late_stage_kernel_size, 1, late_stage_padding, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
                                     0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


# Multiple 5-channel residual block that uses learned switching to diversify its outputs.
# If multi_head_input=False: takes standard (b,f,w,h) input tensor; else takes (b,heads,f,w,h) input tensor. Note that the default RDB block does not support this format, so use SwitchedRDB_5C_MultiHead for this case.
# If collapse_heads=True, outputs (b,f,w,h) tensor.
# If collapse_heads=False, outputs (b,heads,f,w,h) tensor.
class SwitchedRDB_5C(switched_conv.MultiHeadSwitchedAbstractBlock):
    def __init__(self, nf=64, gc=32, num_convs=8, num_heads=2, include_skip_head=False, init_temperature=1, multi_head_input=False, collapse_heads=True, force_block=None):
        if force_block is None:
            rdb5c = functools.partial(ResidualDenseBlock_5C, nf, gc)
        else:
            rdb5c = force_block
        super(SwitchedRDB_5C, self).__init__(
            rdb5c,
            nf,
            num_convs,
            num_heads,
            att_kernel_size=3,
            att_pads=1,
            include_skip_head=include_skip_head,
            initial_temperature=init_temperature,
            multi_head_input=multi_head_input,
            concat_heads_into_filters=collapse_heads,
        )
        self.collapse_heads = collapse_heads
        if self.collapse_heads:
            self.mhead_collapse = nn.Conv2d(num_heads * nf, nf, 1)
            arch_util.initialize_weights([self.mhead_collapse], 1)

        arch_util.initialize_weights([sw.attention_conv1 for sw in self.switches] +
                                     [sw.attention_conv2 for sw in self.switches], 1)

    def forward(self, x, output_attention_weights=False):
        outs = super(SwitchedRDB_5C, self).forward(x, output_attention_weights)
        if output_attention_weights:
            outs, atts = outs

        if self.collapse_heads:
            # outs need to be collapsed back down to a single heads worth of data.
            out = self.mhead_collapse(outs)
        else:
            out = outs

        return out, atts


# Implementation of ResidualDenseBlock_5C which compresses multiple switching heads via a Conv3d before doing RDB
# computation.
class ResidualDenseBlock_5C_WithMheadConverter(ResidualDenseBlock_5C):
    def __init__(self, nf=64, gc=32, bias=True, heads=2):
        # Switched blocks generally operate at low resolution, kernel size is much less important, therefore set to 1.
        super(ResidualDenseBlock_5C_WithMheadConverter, self).__init__(nf=nf, gc=gc, bias=bias, late_stage_kernel_size=1,
                                                                       late_stage_padding=0)
        self.heads = heads
        self.converter = nn.Conv3d(nf, nf, kernel_size=(heads, 1, 1), stride=(heads, 1, 1))
        arch_util.initialize_weights(self.converter)

    # Accepts input of shape (b, heads, f, w, h)
    def forward(self, x):
        # Permute filter dim to 1.
        x = x.permute(0, 2, 1, 3, 4)
        x = self.converter(x)
        x = torch.squeeze(x, dim=2)
        return super(ResidualDenseBlock_5C_WithMheadConverter, self).forward(x)


# Multiple 5-channel residual block that uses learned switching to diversify its outputs. The difference between this
# block and SwitchedRDB_5C is this block accepts multi-headed inputs of format (b,heads,f,w,h).
#
# It does this by performing a Conv3d on the first block, which convolves all heads and collapses them to a dimension
# of 1. The tensor is then squeezed and performs identically to SwitchedRDB_5C from there.
class SwitchedRDB_5C_MultiHead(SwitchedRDB_5C):
    def __init__(self, nf=64, gc=32, num_convs=8, num_heads=2, include_skip_head=False, init_temperature=1, collapse_heads=False):
        rdb5c = functools.partial(ResidualDenseBlock_5C_WithMheadConverter, nf, gc, heads=num_heads)
        super(SwitchedRDB_5C_MultiHead, self).__init__(
            nf=nf,
            gc=gc,
            num_convs=num_convs,
            num_heads=num_heads,
            include_skip_head=include_skip_head,
            init_temperature=init_temperature,
            multi_head_input=True,
            collapse_heads=collapse_heads,
            force_block=rdb5c,
        )


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class LowDimRRDB(RRDB):
    def __init__(self, nf, gc=32, dimensional_adjustment=4):
        super(LowDimRRDB, self).__init__(nf * (dimensional_adjustment ** 2), gc * (dimensional_adjustment ** 2))
        self.unshuffle = PixelUnshuffle(dimensional_adjustment)
        self.shuffle = nn.PixelShuffle(dimensional_adjustment)

    def forward(self, x):
        x = self.unshuffle(x)
        x = super(LowDimRRDB, self).forward(x)
        return self.shuffle(x)


# RRDB block that uses switching on the individual RDB modules that compose it to increase learning diversity.
class SwitchedRRDB(RRDB):
    def __init__(self, nf, gc=32, num_convs=8, init_temperature=1, final_temperature_step=1, switching_block=SwitchedRDB_5C):
        super(SwitchedRRDB, self).__init__(nf, gc)
        self.RDB1 = switching_block(nf, gc, num_convs=num_convs, init_temperature=init_temperature)
        self.RDB2 = switching_block(nf, gc, num_convs=num_convs, init_temperature=init_temperature)
        self.RDB3 = switching_block(nf, gc, num_convs=num_convs, init_temperature=init_temperature)
        self.init_temperature = init_temperature
        self.final_temperature_step = final_temperature_step
        self.running_mean = 0
        self.running_count = 0

    def set_temperature(self, temp):
        [sw.set_attention_temperature(temp) for sw in self.RDB1.switches]
        [sw.set_attention_temperature(temp) for sw in self.RDB2.switches]
        [sw.set_attention_temperature(temp) for sw in self.RDB3.switches]

    def forward(self, x):
        out, att1 = self.RDB1(x, True)
        out, att2 = self.RDB2(out, True)
        out, att3 = self.RDB3(out, True)

        a1mean, _ = switched_conv.compute_attention_specificity(att1, 2)
        a2mean, _ = switched_conv.compute_attention_specificity(att2, 2)
        a3mean, _ = switched_conv.compute_attention_specificity(att3, 2)
        self.running_mean += (a1mean + a2mean + a3mean) / 3.0
        self.running_count += 1

        return out * 0.2 + x

    def get_debug_values(self, step, prefix):
        # Take the chance to update the temperature here.
        temp = max(1, int(self.init_temperature * (self.final_temperature_step - step) / self.final_temperature_step))
        self.set_temperature(temp)

        # Intentionally overwrite attention_temperature from other RRDB blocks; these should be synced.
        val = {"%s_attention_mean" % (prefix,): self.running_mean / self.running_count,
               "attention_temperature": temp}
        self.running_count = 0
        self.running_mean = 0
        return val


# Identical to LowDimRRDB but wraps an RRDB rather than inheriting from it. TODO: remove LowDimRRDB when backwards
# compatibility is no longer desired.
class LowDimRRDBWrapper(nn.Module):
    # Do not specify nf or gc on the partial_rrdb passed in. That will be done by the wrapper.
    def __init__(self, nf, partial_rrdb, gc=32, dimensional_adjustment=4):
        super(LowDimRRDBWrapper, self).__init__()
        self.rrdb = partial_rrdb(nf=nf * (dimensional_adjustment ** 2), gc=gc * (dimensional_adjustment ** 2))
        self.unshuffle = PixelUnshuffle(dimensional_adjustment)
        self.shuffle = nn.PixelShuffle(dimensional_adjustment)

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.rrdb(x)
        return self.shuffle(x)

# RRDB block that uses multi-headed switching on multiple individual RDB blocks to improve diversity. Multiple heads
# are annealed internally. This variant has a depth of 4 RDB blocks, rather than 3 like others above.
class SwitchedMultiHeadRRDB(SwitchedRRDB):
    def __init__(self, nf, gc=32, num_convs=8, num_heads=2, init_temperature=1, final_temperature_step=1):
        super(SwitchedMultiHeadRRDB, self).__init__(nf=nf, gc=gc, num_convs=num_convs, init_temperature=init_temperature, final_temperature_step=final_temperature_step)
        self.RDB1 = SwitchedRDB_5C(nf, gc, num_convs=num_convs, num_heads=num_heads, include_skip_head=True, init_temperature=init_temperature, collapse_heads=False)
        self.RDB2 = SwitchedRDB_5C_MultiHead(nf, gc, num_convs=num_convs, num_heads=num_heads, include_skip_head=True, init_temperature=init_temperature, collapse_heads=False)
        self.RDB3 = SwitchedRDB_5C_MultiHead(nf, gc, num_convs=num_convs, num_heads=num_heads, include_skip_head=True, init_temperature=init_temperature, collapse_heads=False)
        self.RDB4 = SwitchedRDB_5C_MultiHead(nf, gc, num_convs=num_convs, num_heads=num_heads, include_skip_head=True, init_temperature=init_temperature, collapse_heads=True)

    def set_temperature(self, temp):
        [sw.set_attention_temperature(temp) for sw in self.RDB1.switches]
        [sw.set_attention_temperature(temp) for sw in self.RDB2.switches]
        [sw.set_attention_temperature(temp) for sw in self.RDB3.switches]
        [sw.set_attention_temperature(temp) for sw in self.RDB4.switches]

    def forward(self, x):
        out, att1 = self.RDB1(x, True)
        out, att2 = self.RDB2(out, True)
        out, att3 = self.RDB3(out, True)
        out, att4 = self.RDB4(out, True)

        a1mean, _ = switched_conv.compute_attention_specificity(att1, 2)
        a2mean, _ = switched_conv.compute_attention_specificity(att2, 2)
        a3mean, _ = switched_conv.compute_attention_specificity(att3, 2)
        a4mean, _ = switched_conv.compute_attention_specificity(att4, 2)
        self.running_mean += (a1mean + a2mean + a3mean + a4mean) / 3.0
        self.running_count += 1

        return out * 0.2 + x

# This module performs the majority of the processing done by RRDBNet. It just doesn't have the upsampling at the end.
class RRDBTrunk(nn.Module):
    def __init__(self, nf_in, nf_out, nb, gc=32, initial_stride=1, rrdb_block_f=None, conv_first_block=None):
        super(RRDBTrunk, self).__init__()
        if rrdb_block_f is None:
            rrdb_block_f = functools.partial(RRDB, nf=nf_out, gc=gc)

        if conv_first_block is None:
            self.conv_first = nn.Conv2d(nf_in, nf_out, 7, initial_stride, padding=3, bias=True)
        else:
            self.conv_first = conv_first_block

        self.RRDB_trunk, self.rrdb_layers = arch_util.make_layer(rrdb_block_f, nb, True)
        self.trunk_conv = nn.Conv2d(nf_out, nf_out, 3, 1, 1, bias=True)

    # Sets the softmax temperature of each RRDB layer. Only works if you are using attentive
    # convolutions.
    def set_temperature(self, temp):
        for layer in self.rrdb_layers:
            layer.set_temperature(temp)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        return fea

    def get_debug_values(self, step, prefix):
        val = {}
        i = 0
        for block in self.RRDB_trunk._modules.values():
            if hasattr(block, "get_debug_values"):
                val.update(block.get_debug_values(step, "%s_rdb_%i" % (prefix, i)))
                i += 1
        return val

# Adds some base methods that all RRDB* classes will use.
class RRDBBase(nn.Module):
    def __init__(self):
        super(RRDBBase, self).__init__()

    # Sets the softmax temperature of each RRDB layer. Only works if you are using attentive
    # convolutions.
    def set_temperature(self, temp):
        for trunk in self.trunks:
            for layer in trunk.rrdb_layers:
                layer.set_temperature(temp)

    def get_debug_values(self, step):
        val = {}
        for i, trunk in enumerate(self.trunks):
            for j, block in enumerate(trunk.RRDB_trunk._modules.values()):
                if hasattr(block, "get_debug_values"):
                    val.update(block.get_debug_values(step, "trunk_%i_block_%i" % (i, j)))
        return val


# This class uses a RRDBTrunk to perform processing on an image, then upsamples it.
class RRDBNet(RRDBBase):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=2, initial_stride=1,
                 rrdb_block_f=None):
        super(RRDBNet, self).__init__()

        # Trunk - does actual processing.
        self.trunk = RRDBTrunk(in_nc, nf, nb, gc, initial_stride, rrdb_block_f)
        self.trunks = [self.trunk]

        # Upsampling
        self.scale = scale
        self.upconv1 = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.trunk(x)

        if self.scale >= 2:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = self.lrelu(self.upconv1(fea))
        if self.scale >= 4:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = self.lrelu(self.upconv2(fea))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return (out,)

    def load_state_dict(self, state_dict, strict=True):
        # The parameters in self.trunk used to be in this class. To support loading legacy saves, restore them.
        t_state = self.trunk.state_dict()
        for k in t_state.keys():
            if k in state_dict.keys():
                state_dict["trunk.%s" % (k,)] = state_dict.pop(k)
        super(RRDBNet, self).load_state_dict(state_dict, strict)

# Variant of RRDBNet that is "assisted" by an external pretrained image classifier whose
# intermediate layers have been splayed out, pixel-shuffled, and fed back in.
# TODO: Convert to use new RRDBBase hierarchy.
class AssistedRRDBNet(nn.Module):
    # in_nc=number of input channels.
    # out_nc=number of output channels.
    # nf=internal filter count
    # nb=number of additional blocks after the assistance layers.
    # gc=growth channel inside of residual blocks
    # scale=the number of times the output is doubled in size.
    # initial_stride=the stride on the first conv. can be used to downsample the image for processing.
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=2, initial_stride=1):
        super(AssistedRRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 7, initial_stride, padding=3, bias=True)

        # Set-up the assist-net, which should do feature extraction for us.
        self.assistnet = torchvision.models.wide_resnet50_2(pretrained=True)
        self.set_enable_assistnet_training(False)
        assist_nf = [4, 8, 16]  # Fixed for resnet. Re-evaluate if using other networks.
        self.assist2 = RRDB(nf + assist_nf[0], gc)
        self.assist3 = RRDB(nf + sum(assist_nf[:2]), gc)
        self.assist4 = RRDB(nf + sum(assist_nf), gc)
        nf = nf + sum(assist_nf)

        # After this, it's just a "standard" RRDB net.
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.RRDB_trunk = arch_util.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 5, 1, padding=2, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def set_enable_assistnet_training(self, en):
        for p in self.assistnet.parameters():
            p.requires_grad = en

    def res_extract(self, x):
        # Width and height must be factors of 16 to use this architecture. Check that here.
        (b, f, w, h) = x.shape
        assert w % 16 == 0
        assert h % 16 == 0

        x = self.assistnet.conv1(x)
        x = self.assistnet.bn1(x)
        x = self.assistnet.relu(x)
        x = self.assistnet.maxpool(x)

        x = self.assistnet.layer1(x)
        l1 = F.pixel_shuffle(x, 4)
        x = self.assistnet.layer2(x)
        l2 = F.pixel_shuffle(x, 8)
        x = self.assistnet.layer3(x)
        l3 = F.pixel_shuffle(x, 16)
        return l1, l2, l3

    def forward(self, x):
        # Invoke the assistant net first.
        l1, l2, l3 = self.res_extract(x)

        fea = self.conv_first(x)
        fea = self.assist2(torch.cat([fea, l3], dim=1))
        fea = self.assist3(torch.cat([fea, l2], dim=1))
        fea = self.assist4(torch.cat([fea, l1], dim=1))

        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        if self.scale >= 2:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = self.lrelu(self.upconv1(fea))
        if self.scale >= 4:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = self.lrelu(self.upconv2(fea))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return (out,)

class PixShuffleInitialConv(nn.Module):
    def __init__(self, reduction_factor, nf_out):
        super(PixShuffleInitialConv, self).__init__()
        self.conv = nn.Conv2d(3 * (reduction_factor ** 2), nf_out, 1)
        self.unshuffle = PixelUnshuffle(reduction_factor)

    def forward(self, x):
        (b, f, w, h) = x.shape
        # This module can only be applied to input images (with 3 channels)
        assert f == 3

        x = self.unshuffle(x)
        return self.conv(x)

# This class uses a RRDBTrunk to perform processing on an image, then upsamples it.
class PixShuffleRRDB(RRDBBase):
    def __init__(self, nf, nb, gc=32, scale=2, rrdb_block_f=None):
        super(PixShuffleRRDB, self).__init__()

        # This class does a 4x pixel shuffle on the filter count inside the trunk, so nf must be divisible by 16.
        assert nf % 16 == 0

        # Trunk - does actual processing.
        self.trunk = RRDBTrunk(3, nf, nb, gc, 1, rrdb_block_f, PixShuffleInitialConv(4, nf))
        self.trunks = [self.trunk]

        # Upsampling
        pix_nf = int(nf/16)
        self.scale = scale
        self.upconv1 = nn.Conv2d(pix_nf, pix_nf, 5, 1, padding=2, bias=True)
        self.upconv2 = nn.Conv2d(pix_nf, pix_nf, 5, 1, padding=2, bias=True)
        self.HRconv = nn.Conv2d(pix_nf, pix_nf, 5, 1, padding=2, bias=True)
        self.conv_last = nn.Conv2d(pix_nf, 3, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.trunk(x)
        fea = self.pixel_shuffle(fea)

        if self.scale >= 2:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = self.lrelu(self.upconv1(fea))
        if self.scale >= 4:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = self.lrelu(self.upconv2(fea))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return (out,)