import functools
import os

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.checkpoint import checkpoint

from models.archs import SPSR_util as B
from models.archs.SwitchedResidualGenerator_arch import ConfigurableSwitchComputer, ReferenceImageBranch, \
    QueryKeyMultiplexer, QueryKeyPyramidMultiplexer
from models.archs.arch_util import ConvGnLelu, UpconvBlock, MultiConvBlock, ReferenceJoinBlock
from switched_conv import compute_attention_specificity
from switched_conv_util import save_attention_to_image_rgb
from .RRDBNet_arch import RRDB


class ImageGradient(nn.Module):
    def __init__(self):
        super(ImageGradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


class ImageGradientNoPadding(nn.Module):
    def __init__(self):
        super(ImageGradientNoPadding, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)
        

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim = 1)
        
        return x


####################
# Generator
####################


class SPSRNetSimplified(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4):
        super(SPSRNetSimplified, self).__init__()
        n_upscale = int(math.log(upscale, 2))

        # Feature branch
        self.model_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False)
        self.model_shortcut_blk = nn.Sequential(*[RRDB(nf, gc=32) for _ in range(nb)])
        self.feature_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False)
        self.model_upsampler = nn.Sequential(*[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=False, bias=False) for _ in range(n_upscale)])
        self.feature_hr_conv1 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=False)
        self.feature_hr_conv2 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=False)

        # Grad branch
        self.get_g_nopadding = ImageGradientNoPadding()
        self.b_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False, bias=False)
        self.b_concat_decimate_1 = ConvGnLelu(2 * nf, nf, kernel_size=1, norm=False, activation=False, bias=False)
        self.b_proc_block_1 = RRDB(nf, gc=32)
        self.b_concat_decimate_2 = ConvGnLelu(2 * nf, nf, kernel_size=1, norm=False, activation=False, bias=False)
        self.b_proc_block_2 = RRDB(nf, gc=32)
        self.b_concat_decimate_3 = ConvGnLelu(2 * nf, nf, kernel_size=1, norm=False, activation=False, bias=False)
        self.b_proc_block_3 = RRDB(nf, gc=32)
        self.b_concat_decimate_4 = ConvGnLelu(2 * nf, nf, kernel_size=1, norm=False, activation=False, bias=False)
        self.b_proc_block_4 = RRDB(nf, gc=32)

        # Upsampling
        self.grad_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=False)
        b_upsampler = nn.Sequential(*[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=False, bias=False) for _ in range(n_upscale)])
        grad_hr_conv1 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=False)
        grad_hr_conv2 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=False)
        self.branch_upsample = B.sequential(*b_upsampler, grad_hr_conv1, grad_hr_conv2)
        # Conv used to output grad branch shortcut.
        self.grad_branch_output_conv = ConvGnLelu(nf, out_nc, kernel_size=1, norm=False, activation=False, bias=False)

        # Conjoin branch.
        # Note: "_branch_pretrain" is a special tag used to denote parameters that get pretrained before the rest.
        self._branch_pretrain_concat = ConvGnLelu(nf * 2, nf, kernel_size=1, norm=False, activation=False, bias=False)
        self._branch_pretrain_block = RRDB(nf * 2, gc=32)
        self._branch_pretrain_HR_conv0 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=False)
        self._branch_pretrain_HR_conv1 = ConvGnLelu(nf, out_nc, kernel_size=3, norm=False, activation=False, bias=False)

    def forward(self, x):

        x_grad = self.get_g_nopadding(x)
        x = self.model_fea_conv(x)

        x_ori = x
        for i in range(5):
            x = self.model_shortcut_blk[i](x)
        x_fea1 = x

        for i in range(5):
            x = self.model_shortcut_blk[i + 5](x)
        x_fea2 = x

        for i in range(5):
            x = self.model_shortcut_blk[i + 10](x)
        x_fea3 = x

        for i in range(5):
            x = self.model_shortcut_blk[i + 15](x)
        x_fea4 = x

        x = self.model_shortcut_blk[20:](x)
        x = self.feature_lr_conv(x)

        # short cut
        x = x_ori + x
        x = self.model_upsampler(x)
        x = self.feature_hr_conv1(x)
        x = self.feature_hr_conv2(x)

        x_b_fea = self.b_fea_conv(x_grad)
        x_cat_1 = torch.cat([x_b_fea, x_fea1], dim=1)

        x_cat_1 = self.b_concat_decimate_1(x_cat_1)
        x_cat_1 = self.b_proc_block_1(x_cat_1)

        x_cat_2 = torch.cat([x_cat_1, x_fea2], dim=1)

        x_cat_2 = self.b_concat_decimate_2(x_cat_2)
        x_cat_2 = self.b_proc_block_2(x_cat_2)

        x_cat_3 = torch.cat([x_cat_2, x_fea3], dim=1)

        x_cat_3 = self.b_concat_decimate_3(x_cat_3)
        x_cat_3 = self.b_proc_block_3(x_cat_3)

        x_cat_4 = torch.cat([x_cat_3, x_fea4], dim=1)

        x_cat_4 = self.b_concat_decimate_4(x_cat_4)
        x_cat_4 = self.b_proc_block_4(x_cat_4)

        x_cat_4 = self.grad_lr_conv(x_cat_4)

        # short cut
        x_cat_4 = x_cat_4 + x_b_fea
        x_branch = self.branch_upsample(x_cat_4)
        x_out_branch = self.grad_branch_output_conv(x_branch)

        ########
        x_branch_d = x_branch
        x__branch_pretrain_cat = torch.cat([x_branch_d, x], dim=1)
        x__branch_pretrain_cat = self._branch_pretrain_block(x__branch_pretrain_cat)
        x_out = self._branch_pretrain_concat(x__branch_pretrain_cat)
        x_out = self._branch_pretrain_HR_conv0(x_out)
        x_out = self._branch_pretrain_HR_conv1(x_out)

        #########
        return x_out_branch, x_out, x_grad

class Spsr5(nn.Module):
    def __init__(self, in_nc, out_nc, nf, xforms=8, upscale=4, multiplexer_reductions=2, init_temperature=10):
        super(Spsr5, self).__init__()
        n_upscale = int(math.log(upscale, 2))

        # switch options
        transformation_filters = nf
        self.transformation_counts = xforms
        multiplx_fn = functools.partial(QueryKeyMultiplexer, transformation_filters, reductions=multiplexer_reductions)
        pretransform_fn = functools.partial(ConvGnLelu, transformation_filters, transformation_filters, norm=False, bias=False, weight_init_factor=.1)
        transform_fn = functools.partial(MultiConvBlock, transformation_filters, int(transformation_filters * 1.5),
                                         transformation_filters, kernel_size=3, depth=3,
                                         weight_init_factor=.1)

        # Feature branch
        self.model_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False)
        self.noise_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.1)
        self.sw1 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=pretransform_fn, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.sw2 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=pretransform_fn, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.feature_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=True, activation=False)
        self.feature_lr_conv2 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=False)

        # Grad branch. Note - groupnorm on this branch is REALLY bad. Avoid it like the plague.
        self.get_g_nopadding = ImageGradientNoPadding()
        self.grad_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False, bias=False)
        self.noise_ref_join_grad = ReferenceJoinBlock(nf, residual_weight_init_factor=.1)
        self.grad_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.3, final_norm=False)
        self.sw_grad = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=pretransform_fn, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts // 2, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.grad_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.grad_lr_conv2 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.upsample_grad = nn.Sequential(*[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=True, bias=False) for _ in range(n_upscale)])
        self.grad_branch_output_conv = ConvGnLelu(nf, out_nc, kernel_size=1, norm=False, activation=False, bias=True)

        # Join branch (grad+fea)
        self.noise_ref_join_conjoin = ReferenceJoinBlock(nf, residual_weight_init_factor=.1)
        self.conjoin_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.3)
        self.conjoin_sw = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=pretransform_fn, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.final_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.upsample = nn.Sequential(*[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=True, bias=True) for _ in range(n_upscale)])
        self.final_hr_conv1 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=True)
        self.final_hr_conv2 = ConvGnLelu(nf, out_nc, kernel_size=3, norm=False, activation=False, bias=False)
        self.switches = [self.sw1, self.sw2, self.sw_grad, self.conjoin_sw]
        self.attentions = None
        self.init_temperature = init_temperature
        self.final_temperature_step = 10000
        self.lr = None

    def forward(self, x, embedding):
        # The attention_maps debugger outputs <x>. Save that here.
        self.lr = x.detach().cpu()

        noise_stds = []

        x_grad = self.get_g_nopadding(x)

        x = self.model_fea_conv(x)
        x1 = x
        x1, a1 = self.sw1(x1, True, identity=x, att_in=(x1, embedding))

        x2 = x1
        x2, nstd = self.noise_ref_join(x2, torch.randn_like(x2))
        x2, a2 = self.sw2(x2, True, identity=x1, att_in=(x2, embedding))
        noise_stds.append(nstd)

        x_grad = self.grad_conv(x_grad)
        x_grad_identity = x_grad
        x_grad, nstd = self.noise_ref_join_grad(x_grad, torch.randn_like(x_grad))
        x_grad, grad_fea_std = self.grad_ref_join(x_grad, x1)
        x_grad, a3 = self.sw_grad(x_grad, True, identity=x_grad_identity, att_in=(x_grad, embedding))
        x_grad = self.grad_lr_conv(x_grad)
        x_grad = self.grad_lr_conv2(x_grad)
        x_grad_out = self.upsample_grad(x_grad)
        x_grad_out = self.grad_branch_output_conv(x_grad_out)
        noise_stds.append(nstd)

        x_out = x2
        x_out, nstd = self.noise_ref_join_conjoin(x_out, torch.randn_like(x_out))
        x_out, fea_grad_std = self.conjoin_ref_join(x_out, x_grad)
        x_out, a4 = self.conjoin_sw(x_out, True, identity=x2, att_in=(x_out, embedding))
        x_out = self.final_lr_conv(x_out)
        x_out = self.upsample(x_out)
        x_out = self.final_hr_conv1(x_out)
        x_out = self.final_hr_conv2(x_out)
        noise_stds.append(nstd)

        self.attentions = [a1, a2, a3, a4]
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
            if step % 500 == 0:
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


# Variant of Spsr5 which uses multiplexer blocks that are not derived from an embedding. Also makes a few "best practices"
# adjustments learned over the past few weeks (no noise, kernel_size=7
class Spsr6(nn.Module):
    def __init__(self, in_nc, out_nc, nf, xforms=8, upscale=4, multiplexer_reductions=3, init_temperature=10):
        super(Spsr6, self).__init__()
        n_upscale = int(math.log(upscale, 2))

        # switch options
        transformation_filters = nf
        self.transformation_counts = xforms
        multiplx_fn = functools.partial(QueryKeyPyramidMultiplexer, transformation_filters, reductions=multiplexer_reductions)
        transform_fn = functools.partial(MultiConvBlock, transformation_filters, int(transformation_filters * 1.5),
                                         transformation_filters, kernel_size=3, depth=3,
                                         weight_init_factor=.1)

        # Feature branch
        self.model_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=7, norm=False, activation=False)
        self.sw1 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.sw2 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.feature_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=True, activation=False)
        self.feature_lr_conv2 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=False)

        # Grad branch. Note - groupnorm on this branch is REALLY bad. Avoid it like the plague.
        self.get_g_nopadding = ImageGradientNoPadding()
        self.grad_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False, bias=False)
        self.grad_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.3, final_norm=False)
        self.sw_grad = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts // 2, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.grad_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.grad_lr_conv2 = ConvGnLelu(nf, nf, kernel_size=1, norm=False, activation=True, bias=True)
        self.upsample_grad = nn.Sequential(*[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=True, bias=False) for _ in range(n_upscale)])
        self.grad_branch_output_conv = ConvGnLelu(nf, out_nc, kernel_size=1, norm=False, activation=False, bias=True)

        # Join branch (grad+fea)
        self.noise_ref_join_conjoin = ReferenceJoinBlock(nf, residual_weight_init_factor=.1)
        self.conjoin_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.3)
        self.conjoin_sw = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.final_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.upsample = nn.Sequential(*[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=True, bias=True) for _ in range(n_upscale)])
        self.final_hr_conv1 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=True)
        self.final_hr_conv2 = ConvGnLelu(nf, out_nc, kernel_size=1, norm=False, activation=False, bias=False)
        self.switches = [self.sw1, self.sw2, self.sw_grad, self.conjoin_sw]
        self.attentions = None
        self.init_temperature = init_temperature
        self.final_temperature_step = 10000
        self.lr = None

    def forward(self, x):
        # The attention_maps debugger outputs <x>. Save that here.
        self.lr = x.detach().cpu()

        x_grad = self.get_g_nopadding(x)

        x = self.model_fea_conv(x)
        x1 = x
        x1, a1 = self.sw1(x1, True, identity=x)

        x2 = x1
        x2, a2 = self.sw2(x2, True, identity=x1)

        x_grad = self.grad_conv(x_grad)
        x_grad_identity = x_grad
        x_grad, grad_fea_std = self.grad_ref_join(x_grad, x1)
        x_grad, a3 = self.sw_grad(x_grad, True, identity=x_grad_identity)
        x_grad = self.grad_lr_conv(x_grad)
        x_grad = self.grad_lr_conv2(x_grad)
        x_grad_out = self.upsample_grad(x_grad)
        x_grad_out = self.grad_branch_output_conv(x_grad_out)

        x_out = x2
        x_out, fea_grad_std = self.conjoin_ref_join(x_out, x_grad)
        x_out, a4 = self.conjoin_sw(x_out, True, identity=x2)
        x_out = self.final_lr_conv(x_out)
        x_out = checkpoint(self.upsample, x_out)
        x_out = checkpoint(self.final_hr_conv1, x_out)
        x_out = self.final_hr_conv2(x_out)

        self.attentions = [a1, a2, a3, a4]
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
            if step % 500 == 0:
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
               "grad_branch_feat_intg_std_dev": self.grad_fea_std,
               "conjoin_branch_grad_intg_std_dev": self.fea_grad_std}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
        return val

# Variant of Spsr7 which uses multiplexer blocks that feed off of a reference embedding. Also computes that embedding.
class Spsr7(nn.Module):
    def __init__(self, in_nc, out_nc, nf, xforms=8, upscale=4, multiplexer_reductions=3, init_temperature=10):
        super(Spsr7, self).__init__()
        n_upscale = int(math.log(upscale, 2))

        # processing the input embedding
        self.reference_embedding = ReferenceImageBranch(nf)

        # switch options
        self.nf = nf
        transformation_filters = nf
        self.transformation_counts = xforms
        multiplx_fn = functools.partial(QueryKeyMultiplexer, transformation_filters, embedding_channels=512, reductions=multiplexer_reductions)
        transform_fn = functools.partial(MultiConvBlock, transformation_filters, int(transformation_filters * 1.5),
                                         transformation_filters, kernel_size=3, depth=3,
                                         weight_init_factor=.1)

        # Feature branch
        self.model_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=7, norm=False, activation=False)
        self.sw1 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.sw1_out = nn.Sequential(ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True),
                                     ConvGnLelu(nf, 3, kernel_size=1, norm=False, activation=False, bias=True))
        self.sw2 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.feature_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=True, activation=False)
        self.feature_lr_conv2 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=False)
        self.sw2_out = nn.Sequential(ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True),
                                     ConvGnLelu(nf, 3, kernel_size=1, norm=False, activation=False, bias=True))

        # Grad branch. Note - groupnorm on this branch is REALLY bad. Avoid it like the plague.
        self.get_g_nopadding = ImageGradientNoPadding()
        self.grad_conv = ConvGnLelu(in_nc, nf, kernel_size=7, norm=False, activation=False, bias=False)
        self.grad_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.3, final_norm=False)

        self.sw_grad = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts // 2, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.grad_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.grad_lr_conv2 = ConvGnLelu(nf, nf, kernel_size=1, norm=False, activation=True, bias=True)
        self.upsample_grad = nn.Sequential(*[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=True, bias=False) for _ in range(n_upscale)])
        self.grad_branch_output_conv = ConvGnLelu(nf, out_nc, kernel_size=1, norm=False, activation=False, bias=True)

        # Join branch (grad+fea)
        self.noise_ref_join_conjoin = ReferenceJoinBlock(nf, residual_weight_init_factor=.1)
        self.conjoin_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.3)
        self.conjoin_sw = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)
        self.final_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.upsample = nn.Sequential(*[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=True, bias=True) for _ in range(n_upscale)])
        self.final_hr_conv1 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=True)
        self.final_hr_conv2 = ConvGnLelu(nf, out_nc, kernel_size=1, norm=False, activation=False, bias=False)
        self.switches = [self.sw1, self.sw2, self.sw_grad, self.conjoin_sw]
        self.attentions = None
        self.init_temperature = init_temperature
        self.final_temperature_step = 10000
        self.lr = None

    def forward(self, x, ref, ref_center):
        # The attention_maps debugger outputs <x>. Save that here.
        self.lr = x.detach().cpu()

        x_grad = self.get_g_nopadding(x)
        ref_code = self.reference_embedding(ref, ref_center)
        ref_embedding = ref_code.view(-1, self.nf * 8, 1, 1).repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)

        x = self.model_fea_conv(x)
        x1 = x
        x1, a1 = self.sw1(x1, True, identity=x, att_in=(x1, ref_embedding))
        s1out = self.sw1_out(x1)

        x2 = x1
        x2, a2 = self.sw2(x2, True, identity=x1, att_in=(x2, ref_embedding))
        s2out = self.sw2_out(x2)

        x_grad = self.grad_conv(x_grad)
        x_grad_identity = x_grad
        x_grad, grad_fea_std = self.grad_ref_join(x_grad, x1)
        x_grad, a3 = self.sw_grad(x_grad, True, identity=x_grad_identity, att_in=(x_grad, ref_embedding))
        x_grad = self.grad_lr_conv(x_grad)
        x_grad = self.grad_lr_conv2(x_grad)
        x_grad_out = self.upsample_grad(x_grad)
        x_grad_out = self.grad_branch_output_conv(x_grad_out)

        x_out = x2
        x_out, fea_grad_std = self.conjoin_ref_join(x_out, x_grad)
        x_out, a4 = self.conjoin_sw(x_out, True, identity=x2, att_in=(x_out, ref_embedding))
        x_out = self.final_lr_conv(x_out)
        x_out = checkpoint(self.upsample, x_out)
        x_out = checkpoint(self.final_hr_conv1, x_out)
        x_out = self.final_hr_conv2(x_out)

        self.attentions = [a1, a2, a3, a4]
        self.grad_fea_std = grad_fea_std.detach().cpu()
        self.fea_grad_std = fea_grad_std.detach().cpu()
        return x_grad_out, x_out, s1out, s2out

    def set_temperature(self, temp):
        [sw.set_temperature(temp) for sw in self.switches]

    def update_for_step(self, step, experiments_path='.'):
        if self.attentions:
            temp = max(1, 1 + self.init_temperature *
                       (self.final_temperature_step - step) / self.final_temperature_step)
            self.set_temperature(temp)
            if step % 500 == 0:
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
               "grad_branch_feat_intg_std_dev": self.grad_fea_std,
               "conjoin_branch_grad_intg_std_dev": self.fea_grad_std}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
        return val

