import math
import functools
from models.archs.arch_util import MultiConvBlock, ConvGnLelu, ConvGnSilu, ReferenceJoinBlock
from models.archs.SwitchedResidualGenerator_arch import ConfigurableSwitchComputer, gather_2d
from models.archs.SPSR_arch import ImageGradientNoPadding
from torch import nn
import torch
import torch.nn.functional as F
from switched_conv_util import save_attention_to_image_rgb
from switched_conv import compute_attention_specificity
import os
import torchvision
from torch.utils.checkpoint import checkpoint

# VGG-style layer with Conv(stride2)->BN->Activation->Conv->BN->Activation
# Doubles the input filter count.
class HalvingProcessingBlock(nn.Module):
    def __init__(self, filters, factor=2):
        super(HalvingProcessingBlock, self).__init__()
        self.bnconv1 = ConvGnSilu(filters, filters, norm=False, bias=False)
        self.bnconv2 = ConvGnSilu(filters, int(filters * factor), kernel_size=1, stride=2, norm=True, bias=False)

    def forward(self, x):
        x = self.bnconv1(x)
        return self.bnconv2(x)


class ExpansionBlock2(nn.Module):
    def __init__(self, filters_in, filters_out=None, block=ConvGnSilu, factor=2):
        super(ExpansionBlock2, self).__init__()
        if filters_out is None:
            filters_out = int(filters_in / factor)
        self.decimate = block(filters_in, filters_out, kernel_size=1, bias=False, activation=True, norm=False)
        self.process_passthrough = block(filters_out, filters_out, kernel_size=3, bias=True, activation=True, norm=False)
        self.conjoin = block(filters_out*2, filters_out*2, kernel_size=1, bias=False, activation=True, norm=False)
        self.reduce = block(filters_out*2, filters_out, kernel_size=1, bias=False, activation=False, norm=True)

    # input is the feature signal with shape  (b, f, w, h)
    # passthrough is the structure signal with shape (b, f/2, w*2, h*2)
    # output is conjoined upsample with shape (b, f/2, w*2, h*2)
    def forward(self, input, passthrough):
        x = F.interpolate(input, scale_factor=2, mode="nearest")
        x = self.decimate(x)
        p = self.process_passthrough(passthrough)
        x = self.conjoin(torch.cat([x, p], dim=1))
        return self.reduce(x)


# Basic convolutional upsampling block that uses interpolate.
class UpconvBlock(nn.Module):
    def __init__(self, filters_in, filters_out=None, block=ConvGnSilu, norm=True, activation=True, bias=False):
        super(UpconvBlock, self).__init__()
        self.reduce = block(filters_in, filters_out, kernel_size=1, bias=False, activation=False, norm=False)
        self.process = block(filters_out, filters_out, kernel_size=3, bias=bias, activation=activation, norm=norm)

    def forward(self, x):
        x = self.reduce(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.process(x)


class QueryKeyMultiplexer(nn.Module):
    def __init__(self, nf, multiplexer_channels, embedding_channels=216, reductions=3):
        super(QueryKeyMultiplexer, self).__init__()

        # Blocks used to create the query
        self.input_process = ConvGnSilu(nf, nf, activation=True, norm=False, bias=True)
        self.embedding_process = ConvGnSilu(embedding_channels, 128, kernel_size=1, activation=True, norm=False, bias=True)
        self.reduction_blocks = nn.ModuleList([HalvingProcessingBlock(int(nf * 1.5 ** i), factor=1.5) for i in range(reductions)])
        reduction_filters = int(nf * 1.5 ** reductions)
        self.processing_blocks = nn.Sequential(
            ConvGnSilu(reduction_filters + 128, reduction_filters + 64, kernel_size=1, activation=True, norm=False, bias=True),
            ConvGnSilu(reduction_filters + 64, reduction_filters, kernel_size=1, activation=True, norm=False, bias=False),
            ConvGnSilu(reduction_filters, reduction_filters, kernel_size=3, activation=True, norm=True, bias=False),
            ConvGnSilu(reduction_filters, reduction_filters, kernel_size=3, activation=True, norm=True, bias=False))
        self.expansion_blocks = nn.ModuleList([ExpansionBlock2(int(reduction_filters // (1.5 ** i)), factor=1.5) for i in range(reductions)])

        # Blocks used to create the key
        self.key_process = ConvGnSilu(nf, nf, kernel_size=1, activation=True, norm=False, bias=False)

        # Postprocessing blocks.
        self.query_key_combine = ConvGnSilu(nf*2, nf, kernel_size=1, activation=True, norm=False, bias=False)
        self.cbl1 = ConvGnSilu(nf, nf // 2, kernel_size=1, norm=True, bias=False, num_groups=4)
        self.cbl2 = ConvGnSilu(nf // 2, 1, kernel_size=1, norm=False, bias=False)

    def forward(self, x, embedding, transformations):
        q = self.input_process(x)
        embedding = self.embedding_process(embedding)
        reduction_identities = []
        for b in self.reduction_blocks:
            reduction_identities.append(q)
            q = b(q)
        q = self.processing_blocks(torch.cat([q, embedding], dim=1))
        for i, b in enumerate(self.expansion_blocks):
            q = b(q, reduction_identities[-i - 1])

        b, t, f, h, w = transformations.shape
        k = transformations.view(b * t, f, h, w)
        k = self.key_process(k)

        q = q.view(b, 1, f, h, w).repeat(1, t, 1, 1, 1).view(b * t, f, h, w)
        v = self.query_key_combine(torch.cat([q, k], dim=1))

        v = self.cbl1(v)
        v = self.cbl2(v)

        return v.view(b, t, h, w)


# Computes a linear latent by performing processing on the reference image and returning the filters of a single point,
# which should be centered on the image patch being processed.
#
# Output is base_filters * 1.5^3.
class ReferenceImageBranch(nn.Module):
    def __init__(self, base_filters=64):
        super(ReferenceImageBranch, self).__init__()
        final_filters = int(base_filters*1.5**3)
        self.features = nn.Sequential(ConvGnSilu(4, base_filters, kernel_size=7, bias=True),
                                      HalvingProcessingBlock(base_filters, factor=1.5),
                                      HalvingProcessingBlock(int(base_filters*1.5), factor=1.5),
                                      HalvingProcessingBlock(int(base_filters*1.5**2), factor=1.5),
                                      ConvGnSilu(final_filters, final_filters, activation=True, norm=True, bias=False))

    # center_point is a [b,2] long tensor describing the center point of where the patch was taken from the reference
    # image.
    def forward(self, x, center_point):
        x = self.features(x)
        return gather_2d(x, center_point // 8)  # Divide by 8 to scale the center_point down.

class SwitchWithReference(nn.Module):
    def __init__(self, nf, num_transforms, init_temperature=10, has_ref=True):
        super(SwitchWithReference, self).__init__()
        self.nf = nf
        self.transformation_counts = num_transforms
        multiplx_fn = functools.partial(QueryKeyMultiplexer, nf)
        transform_fn = functools.partial(MultiConvBlock, nf, int(nf * 1.25), nf, kernel_size=3, depth=4, weight_init_factor=.1)
        if has_ref:
            self.ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.3, final_norm=False, kernel_size=1, depth=2)
        else:
            self.ref_join = None
        self.switch = ConfigurableSwitchComputer(nf, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)

    def forward(self, x, mplex_ref=None, ref=None):
        if self.ref_join is not None:
            branch, ref_std = self.ref_join(x, ref)
            return self.switch(branch, True, identity=x, att_in=(branch, mplex_ref)) + (ref_std,)
        else:
            return self.switch(x, True, identity=x, att_in=(x, mplex_ref))


class SSGr1(nn.Module):
    def __init__(self, in_nc, out_nc, nf, xforms=8, upscale=4, init_temperature=10):
        super(SSGr1, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        self.nf = nf

        # processing the input embedding
        self.reference_embedding = ReferenceImageBranch(nf)

        # Feature branch
        self.model_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False)
        self.sw1 = SwitchWithReference(nf, xforms, init_temperature, has_ref=False)

        # Grad branch. Note - groupnorm on this branch is REALLY bad. Avoid it like the plague.
        self.get_g_nopadding = ImageGradientNoPadding()
        self.grad_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False, bias=False)
        self.sw_grad = SwitchWithReference(nf, xforms // 2, init_temperature, has_ref=True)
        self.grad_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.upsample_grad = UpconvBlock(nf, nf // 2, block=ConvGnLelu, norm=False, activation=True, bias=False)
        self.grad_branch_output_conv = ConvGnLelu(nf // 2, out_nc, kernel_size=1, norm=False, activation=False, bias=True)

        # Join branch (grad+fea)
        self.conjoin_sw = SwitchWithReference(nf, xforms, init_temperature, has_ref=True)
        self.final_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.upsample = UpconvBlock(nf, nf // 2, block=ConvGnLelu, norm=False, activation=True, bias=True)
        self.final_hr_conv1 = ConvGnLelu(nf // 2, nf // 2, kernel_size=3, norm=False, activation=False, bias=True)
        self.final_hr_conv2 = ConvGnLelu(nf // 2, out_nc, kernel_size=3, norm=False, activation=False, bias=False)
        self.switches = [self.sw1.switch, self.sw_grad.switch, self.conjoin_sw.switch]
        self.attentions = None
        self.lr = None
        self.init_temperature = init_temperature
        self.final_temperature_step = 10000

    def forward(self, x, ref, ref_center, save_attentions=True):
        # The attention_maps debugger outputs <x>. Save that here.
        self.lr = x.detach().cpu()

        # If we're not saving attention, we also shouldn't be updating the attention norm. This is because the attention
        # norm should only be getting updates with new data, not recurrent generator sampling.
        for sw in self.switches:
            sw.set_update_attention_norm(save_attentions)

        x_grad = self.get_g_nopadding(x)
        ref_code = checkpoint(self.reference_embedding, ref, ref_center)
        ref_embedding = ref_code.view(-1, ref_code.shape[1], 1, 1).repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)

        x = self.model_fea_conv(x)
        x1, a1 = checkpoint(self.sw1, x, ref_embedding)

        x_grad = self.grad_conv(x_grad)
        x_grad, a3, grad_fea_std = checkpoint(self.sw_grad, x_grad, ref_embedding, x1)
        x_grad = checkpoint(self.grad_lr_conv, x_grad)
        x_grad_out = checkpoint(self.upsample_grad, x_grad)
        x_grad_out = checkpoint(self.grad_branch_output_conv, x_grad_out)

        x_out, a4, fea_grad_std = checkpoint(self.conjoin_sw, x1, ref_embedding, x_grad)
        x_out = checkpoint(self.final_lr_conv, x_out)
        x_out = checkpoint(self.upsample, x_out)
        x_out = checkpoint(self.final_hr_conv2, x_out)

        if save_attentions:
            self.attentions = [a1, a3, a4]
        self.grad_fea_std = grad_fea_std.detach().cpu()
        self.fea_grad_std = fea_grad_std.detach().cpu()
        return x_grad_out, x_out, x_grad

    def set_temperature(self, temp):
        [sw.set_temperature(temp) for sw in self.switches]

    def update_for_step(self, step, experiments_path='.'):
        if self.attentions:
            temp = max(1, 1 + self.init_temperature *
                       (self.final_temperature_step - step) / self.final_temperature_step)
            self.set_temperature(temp)
            if step % 200 == 0:
                output_path = os.path.join(experiments_path, "attention_maps")
                prefix = "amap_%i_a%i_%%i.png"
                [save_attention_to_image_rgb(output_path, self.attentions[i], self.nf, prefix % (step, i), step, output_mag=False) for i in range(len(self.attentions))]
                torchvision.utils.save_image(self.lr, os.path.join(experiments_path, "attention_maps", "amap_%i_base_image.png" % (step,)))


    def get_debug_values(self, step, net_name):
        if self.attentions:
            temp = self.switches[0].switch.temperature
            mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
            means = [i[0] for i in mean_hists]
            hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
            val = {"switch_temperature": temp,
                   "grad_branch_feat_intg_std_dev": self.grad_fea_std,
                   "conjoin_branch_grad_intg_std_dev": self.fea_grad_std}
            for i in range(len(means)):
                val["switch_%i_specificity" % (i,)] = means[i]
                val["switch_%i_histogram" % (i,)] = hists[i]
        return val


class StackedSwitchGenerator(nn.Module):
    def __init__(self, in_nc, out_nc, nf, xforms=8, upscale=4, init_temperature=10):
        super(StackedSwitchGenerator, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        self.nf = nf

        # processing the input embedding
        self.reference_embedding = ReferenceImageBranch(nf)

        # Feature branch
        self.model_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False)
        self.sw1 = SwitchWithReference(nf, xforms, init_temperature, has_ref=False)
        self.sw2 = SwitchWithReference(nf, xforms, init_temperature, has_ref=False)
        self.sw3 = SwitchWithReference(nf, xforms, init_temperature, has_ref=False)
        self.switches = [self.sw1.switch, self.sw2.switch, self.sw3.switch]

        self.final_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.upsample = UpconvBlock(nf, nf // 2, block=ConvGnLelu, norm=False, activation=True, bias=True)
        self.final_hr_conv1 = ConvGnLelu(nf // 2, nf // 2, kernel_size=3, norm=False, activation=False, bias=True)
        self.final_hr_conv2 = ConvGnLelu(nf // 2, out_nc, kernel_size=3, norm=False, activation=False, bias=False)
        self.attentions = None
        self.lr = None
        self.init_temperature = init_temperature
        self.final_temperature_step = 10000

    def forward(self, x, ref, ref_center, save_attentions=True):
        # The attention_maps debugger outputs <x>. Save that here.
        self.lr = x.detach().cpu()

        ref_code = checkpoint(self.reference_embedding, ref, ref_center)
        ref_embedding = ref_code.view(-1, ref_code.shape[1], 1, 1).repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)

        x = self.model_fea_conv(x)
        x1, a1 = checkpoint(self.sw1, x, ref_embedding)
        x2, a2 = checkpoint(self.sw2, x1, ref_embedding)
        x3, a3 = checkpoint(self.sw3, x2, ref_embedding)
        x_out = checkpoint(self.final_lr_conv, x3)
        x_out = checkpoint(self.upsample, x_out)
        x_out = checkpoint(self.final_hr_conv2, x_out)

        if save_attentions:
            self.attentions = [a1, a3, a3]
        return x_out,

    def set_temperature(self, temp):
        [sw.set_temperature(temp) for sw in self.switches]

    def update_for_step(self, step, experiments_path='.'):
        if self.attentions:
            temp = max(1, 1 + self.init_temperature *
                       (self.final_temperature_step - step) / self.final_temperature_step)
            self.set_temperature(temp)
            if step % 200 == 0:
                output_path = os.path.join(experiments_path, "attention_maps")
                prefix = "amap_%i_a%i_%%i.png"
                [save_attention_to_image_rgb(output_path, self.attentions[i], self.nf, prefix % (step, i), step, output_mag=False) for i in range(len(self.attentions))]
                torchvision.utils.save_image(self.lr, os.path.join(experiments_path, "attention_maps", "amap_%i_base_image.png" % (step,)))


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


class SSGDeep(nn.Module):
    def __init__(self, in_nc, out_nc, nf, xforms=8, upscale=4, init_temperature=10):
        super(SSGDeep, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        self.nf = nf

        # processing the input embedding
        self.reference_embedding = ReferenceImageBranch(nf)

        # Feature branch
        self.model_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=7, norm=False, activation=False)
        self.sw1 = SwitchWithReference(nf, xforms, init_temperature, has_ref=False)

        # Grad branch. Note - groupnorm on this branch is REALLY bad. Avoid it like the plague.
        self.get_g_nopadding = ImageGradientNoPadding()
        self.grad_conv = ConvGnLelu(in_nc, nf, kernel_size=7, norm=False, activation=False, bias=False)
        self.sw_grad = SwitchWithReference(nf, xforms // 2, init_temperature, has_ref=True)
        self.grad_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.upsample_grad = UpconvBlock(nf, nf // 2, block=ConvGnLelu, norm=False, activation=True, bias=False)
        self.grad_branch_output_conv = ConvGnLelu(nf // 2, out_nc, kernel_size=1, norm=False, activation=False, bias=True)

        # Join branch (grad+fea)
        self.conjoin_sw = SwitchWithReference(nf, xforms, init_temperature, has_ref=True)
        self.sw3 = SwitchWithReference(nf, xforms, init_temperature, has_ref=False)
        self.sw4 = SwitchWithReference(nf, xforms, init_temperature, has_ref=False)
        self.final_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.upsample = UpconvBlock(nf, nf // 2, block=ConvGnLelu, norm=False, activation=True, bias=True)
        self.final_hr_conv1 = ConvGnLelu(nf // 2, nf // 2, kernel_size=3, norm=False, activation=False, bias=True)
        self.final_hr_conv2 = ConvGnLelu(nf // 2, out_nc, kernel_size=3, norm=False, activation=False, bias=False)
        self.switches = [self.sw1.switch, self.sw_grad.switch, self.conjoin_sw.switch, self.sw3.switch, self.sw4.switch]
        self.attentions = None
        self.lr = None
        self.init_temperature = init_temperature
        self.final_temperature_step = 10000

    def forward(self, x, ref, ref_center, save_attentions=True):
        # The attention_maps debugger outputs <x>. Save that here.
        self.lr = x.detach().cpu()

        # If we're not saving attention, we also shouldn't be updating the attention norm. This is because the attention
        # norm should only be getting updates with new data, not recurrent generator sampling.
        for sw in self.switches:
            sw.set_update_attention_norm(save_attentions)

        x_grad = self.get_g_nopadding(x)
        ref_code = checkpoint(self.reference_embedding, ref, ref_center)
        ref_embedding = ref_code.view(-1, ref_code.shape[1], 1, 1).repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)

        x = self.model_fea_conv(x)
        x1, a1 = checkpoint(self.sw1, x, ref_embedding)

        x_grad = self.grad_conv(x_grad)
        x_grad, a3, grad_fea_std = checkpoint(self.sw_grad, x_grad, ref_embedding, x1)
        x_grad = checkpoint(self.grad_lr_conv, x_grad)
        x_grad_out = checkpoint(self.upsample_grad, x_grad)
        x_grad_out = checkpoint(self.grad_branch_output_conv, x_grad_out)

        x_out, a4, fea_grad_std = checkpoint(self.conjoin_sw, x1, ref_embedding, x_grad)
        x_out, a5 = checkpoint(self.sw3, x_out, ref_embedding)
        x_out, a6 = checkpoint(self.sw4, x_out, ref_embedding)
        x_out = checkpoint(self.final_lr_conv, x_out)
        x_out = checkpoint(self.upsample, x_out)
        x_out = checkpoint(self.final_hr_conv2, x_out)

        if save_attentions:
            self.attentions = [a1, a3, a4, a5, a6]
        self.grad_fea_std = grad_fea_std.detach().cpu()
        self.fea_grad_std = fea_grad_std.detach().cpu()
        return x_grad_out, x_out, x_grad

    def set_temperature(self, temp):
        [sw.set_temperature(temp) for sw in self.switches]

    def update_for_step(self, step, experiments_path='.'):
        if self.attentions:
            temp = max(1, 1 + self.init_temperature *
                       (self.final_temperature_step - step) / self.final_temperature_step)
            self.set_temperature(temp)
            if step % 200 == 0:
                output_path = os.path.join(experiments_path, "attention_maps")
                prefix = "amap_%i_a%i_%%i.png"
                [save_attention_to_image_rgb(output_path, self.attentions[i], self.nf, prefix % (step, i), step, output_mag=False) for i in range(len(self.attentions))]
                torchvision.utils.save_image(self.lr, os.path.join(experiments_path, "attention_maps", "amap_%i_base_image.png" % (step,)))


    def get_debug_values(self, step, net_name):
        temp = self.switches[0].switch.temperature
        mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
        means = [i[0] for i in mean_hists]
        hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
        val = {"switch_temperature": temp,
               "grad_branch_feat_intg_std_dev": self.grad_fea_std,
               "conjoin_branch_grad_intg_std_dev": self.fea_grad_std}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
        return val


class StackedSwitchGenerator5Layer(nn.Module):
    def __init__(self, in_nc, out_nc, nf, xforms=8, upscale=4, init_temperature=10):
        super(StackedSwitchGenerator5Layer, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        self.nf = nf

        # processing the input embedding
        self.reference_embedding = ReferenceImageBranch(nf)

        # Feature branch
        self.model_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False)
        self.sw1 = SwitchWithReference(nf, xforms, init_temperature, has_ref=False)
        self.sw2 = SwitchWithReference(nf, xforms // 2, init_temperature, has_ref=False)
        self.sw3 = SwitchWithReference(nf, xforms // 2, init_temperature, has_ref=False)
        self.sw4 = SwitchWithReference(nf, xforms // 2, init_temperature, has_ref=False)
        self.sw5 = SwitchWithReference(nf, xforms, init_temperature, has_ref=False)
        self.switches = [self.sw1.switch, self.sw2.switch, self.sw3.switch, self.sw4.switch, self.sw5.switch]

        self.final_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.upsample = UpconvBlock(nf, nf // 2, block=ConvGnLelu, norm=False, activation=True, bias=True)
        self.final_hr_conv1 = ConvGnLelu(nf // 2, nf // 2, kernel_size=3, norm=False, activation=False, bias=True)
        self.final_hr_conv2 = ConvGnLelu(nf // 2, out_nc, kernel_size=3, norm=False, activation=False, bias=False)
        self.attentions = None
        self.lr = None
        self.init_temperature = init_temperature
        self.final_temperature_step = 10000

    def forward(self, x, ref, ref_center, save_attentions=True):
        # The attention_maps debugger outputs <x>. Save that here.
        self.lr = x.detach().cpu()

        # If we're not saving attention, we also shouldn't be updating the attention norm. This is because the attention
        # norm should only be getting updates with new data, not recurrent generator sampling.
        for sw in self.switches:
            sw.set_update_attention_norm(save_attentions)

        ref_code = checkpoint(self.reference_embedding, ref, ref_center)
        ref_embedding = ref_code.view(-1, ref_code.shape[1], 1, 1).repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)

        x = self.model_fea_conv(x)
        x1, a1 = checkpoint(self.sw1, x, ref_embedding)
        x2, a2 = checkpoint(self.sw2, x1, ref_embedding)
        x3, a3 = checkpoint(self.sw3, x2, ref_embedding)
        x4, a4 = checkpoint(self.sw4, x3, ref_embedding)
        x5, a5 = checkpoint(self.sw5, x4, ref_embedding)
        x_out = checkpoint(self.final_lr_conv, x5)
        x_out = checkpoint(self.upsample, x_out)
        x_out = checkpoint(self.final_hr_conv2, x_out)

        if save_attentions:
            self.attentions = [a1, a3, a3, a4, a5]
        return x_out,

    def set_temperature(self, temp):
        [sw.set_temperature(temp) for sw in self.switches]

    def update_for_step(self, step, experiments_path='.'):
        if self.attentions:
            temp = max(1, 1 + self.init_temperature *
                       (self.final_temperature_step - step) / self.final_temperature_step)
            self.set_temperature(temp)
            if step % 200 == 0:
                output_path = os.path.join(experiments_path, "attention_maps")
                prefix = "amap_%i_a%i_%%i.png"
                [save_attention_to_image_rgb(output_path, self.attentions[i], self.nf, prefix % (step, i), step,
                                             output_mag=False) for i in range(len(self.attentions))]
                torchvision.utils.save_image(self.lr, os.path.join(experiments_path, "attention_maps",
                                                                   "amap_%i_base_image.png" % (step,)))

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