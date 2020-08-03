import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs import SPSR_util as B
from .RRDBNet_arch import RRDB
from models.archs.arch_util import ConvGnLelu, ExpansionBlock, UpconvBlock


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
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(SPSRNet, self).__init__()

        n_upscale = int(math.log(upscale, 2))

        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
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
        

    def forward(self, x):    

        x_grad = self.get_g_nopadding(x)
        x = self.model[0](x)  

        x, block_list = self.model[1](x)

        x_ori = x
        for i in range(5):
            x = block_list[i](x)
        x_fea1 = x 

        for i in range(5):
            x = block_list[i+5](x)
        x_fea2 = x

        for i in range(5):
            x = block_list[i+10](x)
        x_fea3 = x
        
        for i in range(5):
            x = block_list[i+15](x)
        x_fea4 = x
        
        x = block_list[20:](x)
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

