import math
import functools
from models.archs.arch_util import MultiConvBlock, ConvGnLelu, ConvGnSilu, ReferenceJoinBlock
from models.archs.SwitchedResidualGenerator_arch import ConfigurableSwitchComputer
from models.archs.SPSR_arch import ImageGradientNoPadding
from torch import nn
import torch
import torch.nn.functional as F
from switched_conv_util import save_attention_to_image_rgb
from switched_conv import compute_attention_specificity
import os
import torchvision


# VGG-style layer with Conv(stride2)->BN->Activation->Conv->BN->Activation
# Doubles the input filter count.
class HalvingProcessingBlock(nn.Module):
    def __init__(self, filters):
        super(HalvingProcessingBlock, self).__init__()
        self.bnconv1 = ConvGnSilu(filters, filters * 2, kernel_size=1, stride=2, norm=False, bias=False)
        self.bnconv2 = ConvGnSilu(filters * 2, filters * 2, norm=True, bias=False)

    def forward(self, x):
        x = self.bnconv1(x)
        return self.bnconv2(x)


class ExpansionBlock2(nn.Module):
    def __init__(self, filters_in, filters_out=None, block=ConvGnSilu):
        super(ExpansionBlock2, self).__init__()
        if filters_out is None:
            filters_out = filters_in // 2
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


class SSGMultiplexer(nn.Module):
    def __init__(self, nf, multiplexer_channels, reductions=2):
        super(SSGMultiplexer, self).__init__()

        # Blocks used to create the query
        self.input_process = ConvGnSilu(nf, nf, activation=True, norm=False, bias=True)
        self.embedding_process = ConvGnSilu(256, 256, activation=True, norm=False, bias=True)
        self.reduction_blocks = nn.ModuleList([HalvingProcessingBlock(nf * 2 ** i) for i in range(reductions)])
        reduction_filters = nf * 2 ** reductions
        self.processing_blocks = nn.Sequential(
            ConvGnSilu(reduction_filters + 256, reduction_filters + 128, kernel_size=1, activation=True, norm=False, bias=True),
            ConvGnSilu(reduction_filters + 128, reduction_filters, kernel_size=3, activation=True, norm=True, bias=False))
        self.expansion_blocks = nn.ModuleList([ExpansionBlock2(reduction_filters // (2 ** i)) for i in range(reductions)])

        # Blocks used to create the key
        self.key_process = ConvGnSilu(nf, nf, kernel_size=1, activation=True, norm=False, bias=True)

        # Postprocessing blocks.
        self.query_key_combine = ConvGnSilu(nf*2, nf, kernel_size=1, activation=True, norm=False, bias=False)
        self.cbl1 = ConvGnSilu(nf, nf // 4, kernel_size=1, activation=True, norm=True, bias=False, num_groups=4)
        self.cbl2 = ConvGnSilu(nf // 4, 1, kernel_size=1, activation=False, norm=False, bias=False)

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

class SSGr1(nn.Module):
    def __init__(self, in_nc, out_nc, nf, xforms=8, upscale=4, init_temperature=10):
        super(SSGr1, self).__init__()
        n_upscale = int(math.log(upscale, 2))

        # switch options
        transformation_filters = nf
        self.transformation_counts = xforms
        multiplx_fn = functools.partial(SSGMultiplexer, transformation_filters)
        transform_fn = functools.partial(MultiConvBlock, transformation_filters, int(transformation_filters * 1.25),
                                         transformation_filters, kernel_size=3, depth=3,
                                         weight_init_factor=.1)

        # Feature branch
        self.model_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False)
        self.noise_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.1, kernel_size=1, depth=2)
        self.sw1 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.feature_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=True, activation=False)
        self.feature_lr_conv2 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=False)

        # Grad branch. Note - groupnorm on this branch is REALLY bad. Avoid it like the plague.
        self.get_g_nopadding = ImageGradientNoPadding()
        self.grad_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False, bias=False)
        self.noise_ref_join_grad = ReferenceJoinBlock(nf, residual_weight_init_factor=.1, kernel_size=1, depth=2)
        self.grad_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.3, final_norm=False, kernel_size=1, depth=2)
        self.sw_grad = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts // 2, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.grad_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.upsample_grad = nn.Sequential(*[UpconvBlock(nf, nf // 2, block=ConvGnLelu, norm=False, activation=True, bias=False) for _ in range(n_upscale)])
        self.grad_branch_output_conv = ConvGnLelu(nf // 2, out_nc, kernel_size=1, norm=False, activation=False, bias=True)

        # Join branch (grad+fea)
        self.noise_ref_join_conjoin = ReferenceJoinBlock(nf, residual_weight_init_factor=.1, kernel_size=1, depth=2)
        self.conjoin_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.3, kernel_size=1, depth=2)
        self.conjoin_sw = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.final_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.upsample = nn.Sequential(*[UpconvBlock(nf, 64, block=ConvGnLelu, norm=False, activation=True, bias=True) for _ in range(n_upscale)])
        self.final_hr_conv2 = ConvGnLelu(64, out_nc, kernel_size=3, norm=False, activation=False, bias=False)
        self.switches = [self.sw1, self.sw_grad, self.conjoin_sw]
        self.attentions = None
        self.lr = None
        self.init_temperature = init_temperature
        self.final_temperature_step = 10000

    def forward(self, x, embedding):
        noise_stds = []
        # The attention_maps debugger outputs <x>. Save that here.
        self.lr = x.detach().cpu()

        x_grad = self.get_g_nopadding(x)

        x = self.model_fea_conv(x)
        x1 = x
        x1, a1 = self.sw1(x1, True, identity=x, att_in=(x1, embedding))

        x_grad = self.grad_conv(x_grad)
        x_grad_identity = x_grad
        x_grad, nstd = self.noise_ref_join_grad(x_grad, torch.randn_like(x_grad))
        x_grad, grad_fea_std = self.grad_ref_join(x_grad, x1)
        x_grad, a3 = self.sw_grad(x_grad, True, identity=x_grad_identity, att_in=(x_grad, embedding))
        x_grad = self.grad_lr_conv(x_grad)
        x_grad_out = self.upsample_grad(x_grad)
        x_grad_out = self.grad_branch_output_conv(x_grad_out)
        noise_stds.append(nstd)

        x_out = x1
        x_out, nstd = self.noise_ref_join_conjoin(x_out, torch.randn_like(x_out))
        x_out, fea_grad_std = self.conjoin_ref_join(x_out, x_grad)
        x_out, a4 = self.conjoin_sw(x_out, True, identity=x1, att_in=(x_out, embedding))
        x_out = self.final_lr_conv(x_out)
        x_out = self.upsample(x_out)
        x_out = self.final_hr_conv2(x_out)
        noise_stds.append(nstd)

        self.attentions = [a1, a3, a4]
        self.noise_stds = torch.stack(noise_stds).mean().detach().cpu()
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
                [save_attention_to_image_rgb(output_path, self.attentions[i], self.transformation_counts, prefix % (step, i), step, output_mag=False) for i in range(len(self.attentions))]
                torchvision.utils.save_image(self.lr, os.path.join(experiments_path, "attention_maps", "amap_%i_base_image.png" % (step,)))


    def get_debug_values(self, step, net_name):
        temp = self.switches[0].switch.temperature
        mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
        means = [i[0] for i in mean_hists]
        hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
        val = {"switch_temperature": temp,
               "noise_branch_std_dev": self.noise_stds,
               "grad_branch_feat_intg_std_dev": self.grad_fea_std,
               "conjoin_branch_grad_intg_std_dev": self.fea_grad_std}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
        return val


class SSGMultiplexerNoEmbedding(nn.Module):
    def __init__(self, nf, multiplexer_channels, reductions=2):
        super(SSGMultiplexerNoEmbedding, self).__init__()

        # Blocks used to create the query
        self.input_process = ConvGnSilu(nf, nf, activation=True, norm=False, bias=True)
        self.reduction_blocks = nn.ModuleList([HalvingProcessingBlock(nf * 2 ** i) for i in range(reductions)])
        reduction_filters = nf * 2 ** reductions
        self.processing_blocks = nn.Sequential(
            ConvGnSilu(reduction_filters, reduction_filters, kernel_size=3, activation=True, norm=True, bias=False),
            ConvGnSilu(reduction_filters, reduction_filters, kernel_size=3, activation=True, norm=True, bias=False))
        self.expansion_blocks = nn.ModuleList([ExpansionBlock2(reduction_filters // (2 ** i)) for i in range(reductions)])

        # Blocks used to create the key
        self.key_process = ConvGnSilu(nf, nf, kernel_size=1, activation=True, norm=False, bias=True)

        # Postprocessing blocks.
        self.query_key_combine = ConvGnSilu(nf*2, nf, kernel_size=1, activation=True, norm=False, bias=False)
        self.cbl1 = ConvGnSilu(nf, nf // 4, kernel_size=1, activation=True, norm=True, bias=False, num_groups=4)
        self.cbl2 = ConvGnSilu(nf // 4, 1, kernel_size=1, activation=False, norm=False, bias=False)

    def forward(self, x, transformations):
        q = self.input_process(x)
        reduction_identities = []
        for b in self.reduction_blocks:
            reduction_identities.append(q)
            q = b(q)
        q = self.processing_blocks(q)
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


class SSGNoEmbedding(nn.Module):
        def __init__(self, in_nc, out_nc, nf, xforms=8, upscale=4, init_temperature=10):
            super(SSGNoEmbedding, self).__init__()
            n_upscale = int(math.log(upscale, 2))

            # switch options
            transformation_filters = nf
            self.transformation_counts = xforms
            multiplx_fn = functools.partial(SSGMultiplexerNoEmbedding, transformation_filters, reductions=3)
            transform_fn = functools.partial(MultiConvBlock, transformation_filters, int(transformation_filters * 1.25),
                                             transformation_filters, kernel_size=3, depth=3,
                                             weight_init_factor=.1)

            # Feature branch
            self.model_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False)
            self.noise_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.1, kernel_size=1, depth=2)
            self.sw1 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                  pre_transform_block=None, transform_block=transform_fn,
                                                  attention_norm=True,
                                                  transform_count=self.transformation_counts,
                                                  init_temp=init_temperature,
                                                  add_scalable_noise_to_transforms=False,
                                                  feed_transforms_into_multiplexer=True)
            self.feature_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=True, activation=False)
            self.feature_lr_conv2 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=False)

            # Grad branch. Note - groupnorm on this branch is REALLY bad. Avoid it like the plague.
            self.get_g_nopadding = ImageGradientNoPadding()
            self.grad_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False, bias=False)
            self.noise_ref_join_grad = ReferenceJoinBlock(nf, residual_weight_init_factor=.1, kernel_size=1, depth=2)
            self.grad_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.3, final_norm=False, kernel_size=1,
                                                    depth=2)
            self.sw_grad = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                      pre_transform_block=None, transform_block=transform_fn,
                                                      attention_norm=True,
                                                      transform_count=self.transformation_counts // 2,
                                                      init_temp=init_temperature,
                                                      add_scalable_noise_to_transforms=False,
                                                      feed_transforms_into_multiplexer=True)
            self.grad_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
            self.upsample_grad = nn.Sequential(
                *[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=True, bias=False) for _ in
                  range(n_upscale)])
            self.grad_branch_output_conv = ConvGnLelu(nf, out_nc, kernel_size=1, norm=False, activation=False,
                                                      bias=True)

            # Join branch (grad+fea)
            self.noise_ref_join_conjoin = ReferenceJoinBlock(nf, residual_weight_init_factor=.1, kernel_size=1, depth=2)
            self.conjoin_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.3, kernel_size=1, depth=2)
            self.conjoin_sw = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                         pre_transform_block=None, transform_block=transform_fn,
                                                         attention_norm=True,
                                                         transform_count=self.transformation_counts,
                                                         init_temp=init_temperature,
                                                         add_scalable_noise_to_transforms=False,
                                                         feed_transforms_into_multiplexer=True)
            self.final_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
            self.upsample = nn.Sequential(
                *[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=True, bias=True) for _ in
                  range(n_upscale)])
            self.final_hr_conv2 = ConvGnLelu(nf, out_nc, kernel_size=3, norm=False, activation=False, bias=False)
            self.switches = [self.sw1, self.sw_grad, self.conjoin_sw]
            self.attentions = None
            self.lr = None
            self.init_temperature = init_temperature
            self.final_temperature_step = 10000

        def forward(self, x, *args):
            noise_stds = []
            # The attention_maps debugger outputs <x>. Save that here.
            self.lr = x.detach().cpu()

            x_grad = self.get_g_nopadding(x)

            x = self.model_fea_conv(x)
            x1 = x
            x1, a1 = self.sw1(x1, True, identity=x)

            x_grad = self.grad_conv(x_grad)
            x_grad_identity = x_grad
            x_grad, nstd = self.noise_ref_join_grad(x_grad, torch.randn_like(x_grad))
            x_grad, grad_fea_std = self.grad_ref_join(x_grad, x1)
            x_grad, a3 = self.sw_grad(x_grad, True, identity=x_grad_identity)
            x_grad = self.grad_lr_conv(x_grad)
            x_grad_out = self.upsample_grad(x_grad)
            x_grad_out = self.grad_branch_output_conv(x_grad_out)
            noise_stds.append(nstd)

            x_out = x1
            x_out, nstd = self.noise_ref_join_conjoin(x_out, torch.randn_like(x_out))
            x_out, fea_grad_std = self.conjoin_ref_join(x_out, x_grad)
            x_out, a4 = self.conjoin_sw(x_out, True, identity=x1)
            x_out = self.final_lr_conv(x_out)
            x_out = self.upsample(x_out)
            x_out = self.final_hr_conv2(x_out)
            noise_stds.append(nstd)

            self.attentions = [a1, a3, a4]
            self.noise_stds = torch.stack(noise_stds).mean().detach().cpu()
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
                    [save_attention_to_image_rgb(output_path, self.attentions[i], self.transformation_counts,
                                                 prefix % (step, i), step, output_mag=False) for i in
                     range(len(self.attentions))]
                    torchvision.utils.save_image(self.lr, os.path.join(experiments_path, "attention_maps",
                                                                       "amap_%i_base_image.png" % (step,)))

        def get_debug_values(self, step, net_name):
            temp = self.switches[0].switch.temperature
            mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
            means = [i[0] for i in mean_hists]
            hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
            val = {"switch_temperature": temp,
                   "noise_branch_std_dev": self.noise_stds,
                   "grad_branch_feat_intg_std_dev": self.grad_fea_std,
                   "conjoin_branch_grad_intg_std_dev": self.fea_grad_std}
            for i in range(len(means)):
                val["switch_%i_specificity" % (i,)] = means[i]
                val["switch_%i_histogram" % (i,)] = hists[i]
            return val



class SSGLite(nn.Module):
        def __init__(self, in_nc, out_nc, nf, xforms=8, upscale=4, init_temperature=10):
            super(SSGLite, self).__init__()

            # switch options
            transformation_filters = nf
            self.transformation_counts = xforms
            multiplx_fn = functools.partial(SSGMultiplexerNoEmbedding, transformation_filters, reductions=3)
            transform_fn = functools.partial(MultiConvBlock, transformation_filters, int(transformation_filters * 1.25),
                                             transformation_filters, kernel_size=5, depth=3,
                                             weight_init_factor=.1)

            self.model_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False)
            self.noise_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.1, kernel_size=1, depth=2)
            self.sw1 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                  pre_transform_block=None, transform_block=transform_fn,
                                                  attention_norm=True,
                                                  transform_count=self.transformation_counts,
                                                  init_temp=init_temperature,
                                                  add_scalable_noise_to_transforms=False,
                                                  feed_transforms_into_multiplexer=True)
            self.intermediate_conv = ConvGnLelu(nf, nf, kernel_size=1, norm=True, activation=False)
            self.noise_ref_join_conjoin = ReferenceJoinBlock(nf, residual_weight_init_factor=.1, kernel_size=1, depth=2)
            self.sw2 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                         pre_transform_block=None, transform_block=transform_fn,
                                                         attention_norm=True,
                                                         transform_count=self.transformation_counts,
                                                         init_temp=init_temperature,
                                                         add_scalable_noise_to_transforms=False,
                                                         feed_transforms_into_multiplexer=True)
            self.intermediate_conv2 = ConvGnLelu(nf, nf, kernel_size=1, norm=True, activation=False)
            self.sw3 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                         pre_transform_block=None, transform_block=transform_fn,
                                                         attention_norm=True,
                                                         transform_count=self.transformation_counts,
                                                         init_temp=init_temperature,
                                                         add_scalable_noise_to_transforms=False,
                                                         feed_transforms_into_multiplexer=True)
            self.final_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
            if upscale > 1:
                n_upscale = int(math.log(upscale, 2))
                self.upsample = nn.Sequential(
                    *[UpconvBlock(nf, 64, block=ConvGnLelu, norm=False, activation=True, bias=True) for _ in
                      range(n_upscale)])
            else:
                self.upsample = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True)
            self.final_hr_conv2 = ConvGnLelu(64, out_nc, kernel_size=3, norm=False, activation=False, bias=False)
            self.switches = [self.sw1, self.sw2, self.sw3]
            self.attentions = None
            self.lr = None
            self.init_temperature = init_temperature
            self.final_temperature_step = 10000

        def forward(self, x, *args):
            # The attention_maps debugger outputs <x>. Save that here.
            self.lr = x.detach().cpu()

            x = self.model_fea_conv(x)
            x1, a1 = self.sw1(x, True)
            x1 = self.intermediate_conv(x1)
            x2, a2 = self.sw2(x1, True)
            x2 = self.intermediate_conv2(x2)
            x3, a3 = self.sw3(x2, True)
            x_out = self.final_lr_conv(x3)
            x_out = self.upsample(x_out)
            x_out = self.final_hr_conv2(x_out)
            self.attentions = [a1, a2, a3]
            return x_out

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
                    [save_attention_to_image_rgb(output_path, self.attentions[i], self.transformation_counts,
                                                 prefix % (step, i), step, output_mag=False) for i in
                     range(len(self.attentions))]
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