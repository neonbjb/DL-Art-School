import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs import SPSR_util as B
from .RRDBNet_arch import RRDB
from models.archs.arch_util import ConvGnLelu, UpconvBlock, ConjoinBlock, ConvGnSilu, MultiConvBlock, ReferenceJoinBlock
from models.archs.SwitchedResidualGenerator_arch import ConvBasisMultiplexer, ConfigurableSwitchComputer, ReferencingConvMultiplexer, ReferenceImageBranch, AdaInConvBlock, ProcessingBranchWithStochasticity
from switched_conv_util import save_attention_to_image_rgb
from switched_conv import compute_attention_specificity
import functools
import os


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

class SPSRNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv', bl_inc=5):
        super(SPSRNet, self).__init__()

        self.bl_inc = bl_inc
        n_upscale = int(math.log(upscale, 2))

        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc + 1, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [RRDB(nf, gc=32) for _ in range(nb)]
        
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_block
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        
        self.HR_conv0_new = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv1_new = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
            *upsampler, self.HR_conv0_new)

        self.get_g_nopadding = ImageGradientNoPadding()

        self.b_fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)

        self.b_concat_1 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_block_1 = RRDB(nf*2, gc=32)
        

        self.b_concat_2 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_block_2 = RRDB(nf*2, gc=32)


        self.b_concat_3 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_block_3 = RRDB(nf*2, gc=32)


        self.b_concat_4 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_block_4 = RRDB(nf*2, gc=32)

        self.b_LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_block
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            b_upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            b_upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        
        b_HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        b_HR_conv1 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)

        self.b_module = B.sequential(*b_upsampler, b_HR_conv0, b_HR_conv1)

        self.conv_w = B.conv_block(nf, out_nc, kernel_size=1, norm_type=None, act_type=None)

        # Note: "_branch_pretrain" is a special tag used to denote parameters that get pretrained before the rest.
        self._branch_pretrain_concat = B.conv_block(nf*2, nf, kernel_size=3, norm_type=None, act_type=None)

        self._branch_pretrain_block = RRDB(nf*2, gc=32)

        self._branch_pretrain_HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self._branch_pretrain_HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        

    def forward(self, x: torch.Tensor):
        x_grad = self.get_g_nopadding(x)

        b, f, w, h = x.shape
        x = torch.cat([x, torch.randn(b, 1, w, h, device=x.get_device())], dim=1)
        x = self.model[0](x)  

        x, block_list = self.model[1](x)

        x_ori = x
        for i in range(self.bl_inc):
            x = block_list[i](x)
        x_fea1 = x 

        for i in range(self.bl_inc):
            x = block_list[i+self.bl_inc](x)
        x_fea2 = x

        for i in range(self.bl_inc):
            x = block_list[i+self.bl_inc*2](x)
        x_fea3 = x
        
        for i in range(self.bl_inc):
            x = block_list[i+self.bl_inc*3](x)
        x_fea4 = x
        
        x = block_list[self.bl_inc*4:](x)
        #short cut
        x = x_ori+x
        x= self.model[2:](x)
        x = self.HR_conv1_new(x)

        x_b_fea = self.b_fea_conv(x_grad)
        x_cat_1 = torch.cat([x_b_fea, x_fea1], dim=1)
        
        x_cat_1 = self.b_block_1(x_cat_1)
        x_cat_1 = self.b_concat_1(x_cat_1)

        x_cat_2 = torch.cat([x_cat_1, x_fea2], dim=1)
        
        x_cat_2 = self.b_block_2(x_cat_2)
        x_cat_2 = self.b_concat_2(x_cat_2)

        x_cat_3 = torch.cat([x_cat_2, x_fea3], dim=1)
        
        x_cat_3 = self.b_block_3(x_cat_3)
        x_cat_3 = self.b_concat_3(x_cat_3)

        x_cat_4 = torch.cat([x_cat_3, x_fea4], dim=1)
        
        x_cat_4 = self.b_block_4(x_cat_4)
        x_cat_4 = self.b_concat_4(x_cat_4)

        x_cat_4 = self.b_LR_conv(x_cat_4)

        #short cut
        x_cat_4 = x_cat_4+x_b_fea
        x_branch = self.b_module(x_cat_4)

        x_out_branch = self.conv_w(x_branch)
        ########
        x_branch_d = x_branch
        x__branch_pretrain_cat = torch.cat([x_branch_d, x], dim=1)
        x__branch_pretrain_cat = self._branch_pretrain_block(x__branch_pretrain_cat)
        x_out = self._branch_pretrain_concat(x__branch_pretrain_cat)
        x_out = self._branch_pretrain_HR_conv0(x_out)
        x_out = self._branch_pretrain_HR_conv1(x_out)
        
        #########
        return x_out_branch, x_out, x_grad


class SwitchedSpsr(nn.Module):
    def __init__(self, in_nc, out_nc, nf, xforms=8, upscale=4, init_temperature=10):
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
        self.grad_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=True, activation=True, bias=False)
        self.grad_hr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=False)
        # Conv used to output grad branch shortcut.
        self.grad_branch_output_conv = ConvGnLelu(nf, out_nc, kernel_size=1, norm=False, activation=False, bias=False)

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
        self.upsample = nn.Sequential(*[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=False, bias=False) for _ in range(n_upscale)])
        self.upsample_grad = nn.Sequential(*[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=False, bias=False) for _ in range(n_upscale)])
        self.final_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=True, activation=False)
        self.final_hr_conv1 = ConvGnLelu(nf, nf, kernel_size=3, norm=True, activation=True, bias=False)
        self.final_hr_conv2 = ConvGnLelu(nf, out_nc, kernel_size=3, norm=False, activation=False, bias=False)
        self.switches = [self.sw1, self.sw2, self.sw_grad, self._branch_pretrain_sw]
        self.attentions = None
        self.init_temperature = init_temperature
        self.final_temperature_step = 10000

    def forward(self, x):
        x_grad = self.get_g_nopadding(x)
        x = self.model_fea_conv(x)

        x1, a1 = self.sw1(x, True)
        x2, a2 = self.sw2(x1, True)
        x_fea = self.feature_lr_conv(x2)
        x_fea = self.feature_hr_conv2(x_fea)

        x_b_fea = self.b_fea_conv(x_grad)
        x_grad, a3 = self.sw_grad(x_b_fea, att_in=torch.cat([x1, x_b_fea], dim=1), output_attention_weights=True)
        x_grad = self.grad_lr_conv(x_grad)
        x_grad = self.grad_hr_conv(x_grad)
        x_out_branch = self.upsample_grad(x_grad)
        x_out_branch = self.grad_branch_output_conv(x_out_branch)

        x__branch_pretrain_cat = torch.cat([x_grad, x_fea], dim=1)
        x__branch_pretrain_cat, a4 = self._branch_pretrain_sw(x__branch_pretrain_cat, att_in=x_fea, identity=x_fea, output_attention_weights=True)
        x_out = self.final_lr_conv(x__branch_pretrain_cat)
        x_out = self.upsample(x_out)
        x_out = self.final_hr_conv1(x_out)
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

    def get_debug_values(self, step):
        temp = self.switches[0].switch.temperature
        mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
        means = [i[0] for i in mean_hists]
        hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
        val = {"switch_temperature": temp}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
        return val


class RefJoiner(nn.Module):
    def __init__(self, nf):
        super(RefJoiner, self).__init__()
        self.lin1 = nn.Linear(512, 256)
        self.lin2 = nn.Linear(256, nf)
        self.join = ReferenceJoinBlock(nf, residual_weight_init_factor=.05, norm=False)

    def forward(self, x, ref):
        ref = self.lin1(ref)
        ref = self.lin2(ref)
        b, _, h, w = x.shape
        ref = ref.view(b, -1, 1, 1)
        return self.join(x, ref.repeat((1, 1, h, w)))


class SwitchedSpsrWithRef2(nn.Module):
    def __init__(self, in_nc, out_nc, nf, xforms=8, upscale=4, init_temperature=10):
        super(SwitchedSpsrWithRef2, self).__init__()
        n_upscale = int(math.log(upscale, 2))

        # switch options
        transformation_filters = nf
        switch_filters = nf
        self.transformation_counts = xforms
        multiplx_fn = functools.partial(ConvBasisMultiplexer, transformation_filters, switch_filters, 3,
                                        2, use_exp2=True)
        pretransform_fn = functools.partial(ConvGnLelu, transformation_filters, transformation_filters, norm=False, bias=False, weight_init_factor=.1)
        transform_fn = functools.partial(MultiConvBlock, transformation_filters, int(transformation_filters * 1.5),
                                         transformation_filters, kernel_size=3, depth=3,
                                         weight_init_factor=.1)

        self.reference_processor = ReferenceImageBranch(transformation_filters)

        # Feature branch
        self.model_fea_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False)
        self.noise_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.01, norm=False)
        self.ref_join1 = RefJoiner(nf)
        self.sw1 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=pretransform_fn, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False)
        self.ref_join2 = RefJoiner(nf)
        self.sw2 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=pretransform_fn, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False)
        self.feature_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=True, activation=False)
        self.feature_lr_conv2 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=False)

        # Grad branch. Note - groupnorm on this branch is REALLY bad. Avoid it like the plague.
        self.get_g_nopadding = ImageGradientNoPadding()
        self.grad_conv = ConvGnLelu(in_nc, nf, kernel_size=3, norm=False, activation=False, bias=False)
        self.noise_ref_join_grad = ReferenceJoinBlock(nf, residual_weight_init_factor=.01, norm=False)
        self.ref_join3 = RefJoiner(nf)
        self.grad_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.2, norm=False, final_norm=False)
        self.sw_grad = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=pretransform_fn, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts // 2, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False)
        self.grad_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.grad_lr_conv2 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.upsample_grad = nn.Sequential(*[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=True, bias=False) for _ in range(n_upscale)])
        self.grad_branch_output_conv = ConvGnLelu(nf, out_nc, kernel_size=1, norm=False, activation=False, bias=True)

        # Join branch (grad+fea)
        self.ref_join4 = RefJoiner(nf)
        self.noise_ref_join_conjoin = ReferenceJoinBlock(nf, residual_weight_init_factor=.01, norm=False)
        self.conjoin_ref_join = ReferenceJoinBlock(nf, residual_weight_init_factor=.2, norm=False)
        self.conjoin_sw = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=pretransform_fn, transform_block=transform_fn,
                                                   attention_norm=True,
                                                   transform_count=self.transformation_counts, init_temp=init_temperature,
                                                   add_scalable_noise_to_transforms=False)
        self.final_lr_conv = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=True, bias=True)
        self.upsample = nn.Sequential(*[UpconvBlock(nf, nf, block=ConvGnLelu, norm=False, activation=True, bias=True) for _ in range(n_upscale)])
        self.final_hr_conv1 = ConvGnLelu(nf, nf, kernel_size=3, norm=False, activation=False, bias=True)
        self.final_hr_conv2 = ConvGnLelu(nf, out_nc, kernel_size=3, norm=False, activation=False, bias=False)
        self.switches = [self.sw1, self.sw2, self.sw_grad, self.conjoin_sw]
        self.attentions = None
        self.init_temperature = init_temperature
        self.final_temperature_step = 10000

    def forward(self, x, ref, center_coord):
        x_grad = self.get_g_nopadding(x)
        ref = self.reference_processor(ref, center_coord)

        x = self.model_fea_conv(x)
        x1 = x
        x1 = self.ref_join1(x1, ref)
        x1, a1 = self.sw1(x1, True, identity=x)

        x2 = x1
        x2 = self.noise_ref_join(x2, torch.randn_like(x2))
        x2 = self.ref_join2(x2, ref)
        x2, a2 = self.sw2(x2, True, identity=x1)

        x_grad = self.grad_conv(x_grad)
        x_grad_identity = x_grad
        x_grad = self.noise_ref_join_grad(x_grad, torch.randn_like(x_grad))
        x_grad = self.ref_join3(x_grad, ref)
        x_grad = self.grad_ref_join(x_grad, x1)
        x_grad, a3 = self.sw_grad(x_grad, True, identity=x_grad_identity)
        x_grad = self.grad_lr_conv(x_grad)
        x_grad = self.grad_lr_conv2(x_grad)
        x_grad_out = self.upsample_grad(x_grad)
        x_grad_out = self.grad_branch_output_conv(x_grad_out)

        x_out = x2
        x_out = self.noise_ref_join_conjoin(x_out, torch.randn_like(x_out))
        x_out = self.ref_join4(x_out, ref)
        x_out = self.conjoin_ref_join(x_out, x_grad)
        x_out, a4 = self.conjoin_sw(x_out, True, identity=x2)
        x_out = self.final_lr_conv(x_out)
        x_out = self.upsample(x_out)
        x_out = self.final_hr_conv1(x_out)
        x_out = self.final_hr_conv2(x_out)

        self.attentions = [a1, a2, a3, a4]

        return x_grad_out, x_out, x_grad

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

    def get_debug_values(self, step):
        temp = self.switches[0].switch.temperature
        mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
        means = [i[0] for i in mean_hists]
        hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
        val = {"switch_temperature": temp}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
        return val
