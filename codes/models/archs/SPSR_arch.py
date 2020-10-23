import functools
import os

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.util import checkpoint

from models.archs import SPSR_util as B
from models.archs.SwitchedResidualGenerator_arch import ConfigurableSwitchComputer, ReferenceImageBranch, \
    QueryKeyMultiplexer, QueryKeyPyramidMultiplexer, ConvBasisMultiplexer
from models.archs.arch_util import ConvGnLelu, UpconvBlock, MultiConvBlock, ReferenceJoinBlock
from switched_conv.switched_conv import compute_attention_specificity
from switched_conv.switched_conv_util import save_attention_to_image_rgb
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


# Variant of Spsr6 which uses multiplexer blocks that feed off of a reference embedding. Also computes that embedding.
class Spsr7(nn.Module):
    def __init__(self, in_nc, out_nc, nf, xforms=8, upscale=4, multiplexer_reductions=3, recurrent=False, init_temperature=10):
        super(Spsr7, self).__init__()
        n_upscale = int(math.log(upscale, 2))

        # processing the input embedding
        self.reference_embedding = ReferenceImageBranch(nf)

        self.recurrent = recurrent
        if recurrent:
            self.model_recurrent_conv = ConvGnLelu(3, nf, kernel_size=3, stride=2, norm=False, activation=False,
                                                   bias=True)
            self.model_fea_recurrent_combine = ConvGnLelu(nf * 2, nf, 1, activation=False, norm=False, bias=False, weight_init_factor=.01)

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
        self.sw2 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False, feed_transforms_into_multiplexer=True)

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

    def forward(self, x, ref, ref_center, update_attention_norm=True, recurrent=None):
        # The attention_maps debugger outputs <x>. Save that here.
        self.lr = x.detach().cpu()

        x_grad = self.get_g_nopadding(x)
        ref_code = self.reference_embedding(ref, ref_center)
        ref_embedding = ref_code.view(-1, self.nf * 8, 1, 1).repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)

        x = self.model_fea_conv(x)
        if self.recurrent:
            rec = self.model_recurrent_conv(recurrent)
            br = self.model_fea_recurrent_combine(torch.cat([x, rec], dim=1))
            x = x + br

        x1 = x
        x1, a1 = self.sw1(x1, identity=x, att_in=(x1, ref_embedding), do_checkpointing=True)

        x2 = x1
        x2, a2 = self.sw2(x2, identity=x1, att_in=(x2, ref_embedding), do_checkpointing=True)

        x_grad = self.grad_conv(x_grad)
        x_grad_identity = x_grad
        x_grad, grad_fea_std = checkpoint(self.grad_ref_join, x_grad, x1)
        x_grad, a3 = self.sw_grad(x_grad, identity=x_grad_identity, att_in=(x_grad, ref_embedding), do_checkpointing=True)
        x_grad = checkpoint(self.grad_lr_conv, x_grad)
        x_grad = checkpoint(self.grad_lr_conv2, x_grad)
        x_grad_out = checkpoint(self.upsample_grad, x_grad)
        x_grad_out = checkpoint(self.grad_branch_output_conv, x_grad_out)

        x_out = x2
        x_out, fea_grad_std = self.conjoin_ref_join(x_out, x_grad)
        x_out, a4 = self.conjoin_sw(x_out, identity=x2, att_in=(x_out, ref_embedding), do_checkpointing=True)
        x_out = checkpoint(self.final_lr_conv, x_out)
        x_out = checkpoint(self.upsample, x_out)
        x_out = checkpoint(self.final_hr_conv1, x_out)
        x_out = checkpoint(self.final_hr_conv2, x_out)

        self.attentions = [a1, a2, a3, a4]
        self.grad_fea_std = grad_fea_std.detach().cpu()
        self.fea_grad_std = fea_grad_std.detach().cpu()
        return x_grad_out, x_out

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


class AttentionBlock(nn.Module):
    def __init__(self, nf, num_transforms, multiplexer_reductions, init_temperature=10, has_ref=True):
        super(AttentionBlock, self).__init__()
        self.nf = nf
        self.transformation_counts = num_transforms
        multiplx_fn = functools.partial(QueryKeyMultiplexer, nf, embedding_channels=512, reductions=multiplexer_reductions)
        transform_fn = functools.partial(MultiConvBlock, nf, int(nf * 1.5),
                                         nf, kernel_size=3, depth=4,
                                         weight_init_factor=.1)
        if has_ref:
            self.ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.3, final_norm=False)
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
            return self.switch(branch, identity=x, att_in=(branch, mplex_ref)) + (ref_std,)
        else:
            return self.switch(x, identity=x, att_in=(x, mplex_ref))


class SwitchedSpsr(nn.Module):
    def __init__(self, in_nc, nf, xforms=8, upscale=4, init_temperature=10):
        super(SwitchedSpsr, self).__init__()
        n_upscale = int(math.log(upscale, 2))

        # switch options
        transformation_filters = nf
        switch_filters = nf
        switch_reductions = 3
        switch_processing_layers = 2
        self.transformation_counts = xforms
        multiplx_fn = functools.partial(ConvBasisMultiplexer, transformation_filters, switch_filters, switch_reductions,
                                        switch_processing_layers, self.transformation_counts, use_exp2=True)
        pretransform_fn = functools.partial(ConvGnLelu, transformation_filters, transformation_filters, norm=False, bias=False, weight_init_factor=.1)
        transform_fn = functools.partial(MultiConvBlock, transformation_filters, int(transformation_filters * 1.5),
                                         transformation_filters, kernel_size=3, depth=3,
                                         weight_init_factor=.1)

        # Feature branch
        self.model_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False)
        self.sw1 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=pretransform_fn, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=True)
        self.sw2 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=pretransform_fn, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=True)
        self.feature_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=True, activation=False)
        self.feature_hr_conv2 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=False)

        # Grad branch
        self.get_g_nopadding = ImageGradientNoPadding()
        self.b_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False, bias=False)
        mplex_grad = functools.partial(ConvBasisMultiplexer, nf * 2, nf * 2, switch_reductions,
                                        switch_processing_layers, self.transformation_counts // 2, use_exp2=True)
        self.sw_grad = ConfigurableSwitchComputer(transformation_filters, mplex_grad,
                                                   pre_transform_block=pretransform_fn, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts // 2, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=True)
        # Upsampling
        self.grad_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=False)
        self.grad_hr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=False)
        # Conv used to output grad branch shortcut.
        self.grad_branch_output_conv = ConvGnLelu(nf, 3, kernel_size=1, norm=False, activation=False, bias=False)

        # Conjoin branch.
        # Note: "_branch_pretrain" is a special tag used to denote parameters that get pretrained before the rest.
        transform_fn_cat = functools.partial(MultiConvBlock, transformation_filters * 2, int(transformation_filters * 1.5),
                                         transformation_filters, kernel_size=3, depth=4,
                                         weight_init_factor=.1)
        pretransform_fn_cat = functools.partial(ConvGnLelu, transformation_filters * 2, transformation_filters * 2, norm=False, bias=False, weight_init_factor=.1)
        self._branch_pretrain_sw = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=pretransform_fn_cat, transform_block=transform_fn_cat,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=True)
        self.upsample = nn.Sequential(*[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=False, bias=True) for _ in range(n_upscale)])
        self.upsample_grad = nn.Sequential(*[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=False, bias=True) for _ in range(n_upscale)])
        self.final_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=True, activation=False)
        self.final_hr_conv1 = ConvGnLelu(nf, nf, kernel_size=3, norm=True, activation=True, bias=True)
        self.final_hr_conv2 = ConvGnLelu(nf, 3, kernel_size=3, norm=False, activation=False, bias=False)
        self.switches = [self.sw1, self.sw2, self.sw_grad, self._branch_pretrain_sw]
        self.attentions = None
        self.init_temperature = init_temperature
        self.final_temperature_step = 10000

    def forward(self, x):
        x_grad = self.get_g_nopadding(x)
        x = self.model_fea_conv(x)

        x1, a1 = self.sw1(x, do_checkpointing=True)
        x2, a2 = self.sw2(x1, do_checkpointing=True)
        x_fea = self.feature_lr_conv(x2)
        x_fea = self.feature_hr_conv2(x_fea)

        x_b_fea = self.b_fea_conv(x_grad)
        x_grad, a3 = self.sw_grad(x_b_fea, att_in=torch.cat([x1, x_b_fea], dim=1), output_attention_weights=True, do_checkpointing=True)
        x_grad = checkpoint(self.grad_lr_conv, x_grad)
        x_grad = checkpoint(self.grad_hr_conv, x_grad)
        x_out_branch = checkpoint(self.upsample_grad, x_grad)
        x_out_branch = self.grad_branch_output_conv(x_out_branch)

        x__branch_pretrain_cat = torch.cat([x_grad, x_fea], dim=1)
        x__branch_pretrain_cat, a4 = self._branch_pretrain_sw(x__branch_pretrain_cat, att_in=x_fea, identity=x_fea, output_attention_weights=True)
        x_out = checkpoint(self.final_lr_conv, x__branch_pretrain_cat)
        x_out = checkpoint(self.upsample, x_out)
        x_out = checkpoint(self.final_hr_conv1, x_out)
        x_out = self.final_hr_conv2(x_out)

        self.attentions = [a1, a2, a3, a4]

        return x_out_branch, x_out, x_grad

    def set_temperature(self, temp):
        [sw.set_temperature(temp) for sw in self.switches]

    def update_for_step(self, step, experiments_path='.'):
        if self.attentions:
            temp = max(1, 1 + self.init_temperature *
                       (self.final_temperature_step - step) / self.final_temperature_step)
            self.set_temperature(temp)
            if step % 200 == 0:
                output_path = os.path.join(experiments_path, "attention_maps", "a%i")
                prefix = "attention_map_%i_%%i.png" % (step,)
                [save_attention_to_image_rgb(output_path % (i,), self.attentions[i], self.transformation_counts, prefix, step) for i in range(len(self.attentions))]

    def get_debug_values(self, step, net):
        temp = self.switches[0].switch.temperature
        mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
        means = [i[0] for i in mean_hists]
        hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
        val = {"switch_temperature": temp}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
        return val