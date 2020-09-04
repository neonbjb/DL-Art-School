import torch
import logging
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.DiscriminatorResnet_arch as DiscriminatorResnet_arch
import models.archs.DiscriminatorResnet_arch_passthrough as DiscriminatorResnet_arch_passthrough
import models.archs.FlatProcessorNetNew_arch as FlatProcessorNetNew_arch
import models.archs.RRDBNet_arch as RRDBNet_arch
import models.archs.HighToLowResNet as HighToLowResNet
import models.archs.NestedSwitchGenerator as ng
import models.archs.feature_arch as feature_arch
import models.archs.SwitchedResidualGenerator_arch as SwitchedGen_arch
import models.archs.SRG1_arch as srg1
import models.archs.ProgressiveSrg_arch as psrg
import models.archs.SPSR_arch as spsr
import models.archs.arch_util as arch_util
import functools
from collections import OrderedDict

logger = logging.getLogger('base')

# Generator
def define_G(opt, net_key='network_G', scale=None):
    if net_key is not None:
        opt_net = opt[net_key]
    else:
        opt_net = opt
    if scale is None:
        scale = opt['scale']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        # RRDB does scaling in two steps, so take the sqrt of the scale we actually want to achieve and feed it to RRDB.
        initial_stride = 1 if 'initial_stride' not in opt_net else opt_net['initial_stride']
        assert initial_stride == 1 or initial_stride == 2
        # Need to adjust the scale the generator sees by the stride since the stride causes a down-sample.
        gen_scale = scale * initial_stride
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], scale=gen_scale, initial_stride=initial_stride)
    elif which_model == "ConfigurableSwitchedResidualGenerator2":
        netG = SwitchedGen_arch.ConfigurableSwitchedResidualGenerator2(switch_depth=opt_net['switch_depth'], switch_filters=opt_net['switch_filters'],
                                                                      switch_reductions=opt_net['switch_reductions'],
                                                                      switch_processing_layers=opt_net['switch_processing_layers'], trans_counts=opt_net['trans_counts'],
                                                                      trans_kernel_sizes=opt_net['trans_kernel_sizes'], trans_layers=opt_net['trans_layers'],
                                                                      transformation_filters=opt_net['transformation_filters'], attention_norm=opt_net['attention_norm'],
                                                                      initial_temp=opt_net['temperature'], final_temperature_step=opt_net['temperature_final_step'],
                                                                      heightened_temp_min=opt_net['heightened_temp_min'], heightened_final_step=opt_net['heightened_final_step'],
                                                                      upsample_factor=scale, add_scalable_noise_to_transforms=opt_net['add_noise'])
    elif which_model == "ConfigurableSwitchedResidualGenerator4":
        netG = SwitchedGen_arch.ConfigurableSwitchedResidualGenerator4(switch_filters=opt_net['switch_filters'],
                                                                      switch_reductions=opt_net['switch_reductions'],
                                                                      switch_processing_layers=opt_net['switch_processing_layers'], trans_counts=opt_net['trans_counts'],
                                                                      trans_kernel_sizes=opt_net['trans_kernel_sizes'], trans_layers=opt_net['trans_layers'],
                                                                      transformation_filters=opt_net['transformation_filters'], attention_norm=opt_net['attention_norm'],
                                                                      initial_temp=opt_net['temperature'], final_temperature_step=opt_net['temperature_final_step'],
                                                                      heightened_temp_min=opt_net['heightened_temp_min'], heightened_final_step=opt_net['heightened_final_step'],
                                                                      upsample_factor=scale, add_scalable_noise_to_transforms=opt_net['add_noise'])
    elif which_model == 'spsr_net':
        netG = spsr.SPSRNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
                            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv', bl_inc=opt_net['bl_inc'])
        if opt['is_train']:
            arch_util.initialize_weights(netG, scale=.1)
    elif which_model == 'spsr_net_improved':
        netG = spsr.SPSRNetSimplified(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                            nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == "spsr_switched":
        xforms = opt_net['num_transforms'] if 'num_transforms' in opt_net.keys() else 8
        netG = spsr.SwitchedSpsr(in_nc=3, out_nc=3, nf=opt_net['nf'], xforms=xforms, upscale=opt_net['scale'],
                                 init_temperature=opt_net['temperature'] if 'temperature' in opt_net.keys() else 10)
    elif which_model == "spsr_switched_with_ref":
        xforms = opt_net['num_transforms'] if 'num_transforms' in opt_net.keys() else 8
        netG = spsr.SwitchedSpsrWithRef(in_nc=3, out_nc=3, nf=opt_net['nf'], xforms=xforms, upscale=opt_net['scale'],
                                 init_temperature=opt_net['temperature'] if 'temperature' in opt_net.keys() else 10)
    elif which_model == "spsr_switched_with_ref4x":
        xforms = opt_net['num_transforms'] if 'num_transforms' in opt_net.keys() else 8
        netG = spsr.SwitchedSpsrWithRef4x(in_nc=3, out_nc=3, nf=opt_net['nf'], xforms=xforms, upscale=opt_net['scale'],
                                 init_temperature=opt_net['temperature'] if 'temperature' in opt_net.keys() else 10)

    # image corruption
    elif which_model == 'HighToLowResNet':
        netG = HighToLowResNet.HighToLowResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                nf=opt_net['nf'], nb=opt_net['nb'], downscale=opt_net['scale'])
    elif which_model == 'FlatProcessorNet':
        '''netG = FlatProcessorNet_arch.FlatProcessorNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                nf=opt_net['nf'], downscale=opt_net['scale'], reduce_anneal_blocks=opt_net['ra_blocks'],
                                assembler_blocks=opt_net['assembler_blocks'])'''
        netG = FlatProcessorNetNew_arch.fixup_resnet34(num_filters=opt_net['nf'])\

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


class GradDiscWrapper(torch.nn.Module):
    def __init__(self, m):
        super(GradDiscWrapper, self).__init__()
        logger.info("Wrapping a discriminator..")
        self.m = m

    def forward(self, x):
        return self.m(x)

def define_D_net(opt_net, img_sz=None, wrap=False):
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'], input_img_factor=img_sz / 128, extra_conv=opt_net['extra_conv'])
    elif which_model == 'discriminator_vgg_128_gn':
        netD = SRGAN_arch.Discriminator_VGG_128_GN(in_nc=opt_net['in_nc'], nf=opt_net['nf'], input_img_factor=img_sz / 128)
        if wrap:
            netD = GradDiscWrapper(netD)
    elif which_model == 'discriminator_resnet':
        netD = DiscriminatorResnet_arch.fixup_resnet34(num_filters=opt_net['nf'], num_classes=1, input_img_size=img_sz)
    elif which_model == 'discriminator_resnet_passthrough':
        netD = DiscriminatorResnet_arch_passthrough.fixup_resnet34(num_filters=opt_net['nf'], num_classes=1, input_img_size=img_sz,
                                                                   number_skips=opt_net['number_skips'], use_bn=True,
                                                                   disable_passthrough=opt_net['disable_passthrough'])
    elif which_model == 'discriminator_pix':
        netD = SRGAN_arch.Discriminator_VGG_PixLoss(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == "discriminator_unet":
        netD = SRGAN_arch.Discriminator_UNet(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == "discriminator_unet_fea":
        netD = SRGAN_arch.Discriminator_UNet_FeaOut(in_nc=opt_net['in_nc'], nf=opt_net['nf'], feature_mode=opt_net['feature_mode'])
    elif which_model == "discriminator_switched":
        netD = SRGAN_arch.Discriminator_switched(in_nc=opt_net['in_nc'], nf=opt_net['nf'], initial_temp=opt_net['initial_temp'],
                                                    final_temperature_step=opt_net['final_temperature_step'])
    elif which_model == "cross_compare_vgg128":
        netD = SRGAN_arch.CrossCompareDiscriminator(in_nc=opt_net['in_nc'], ref_channels=opt_net['ref_channels'] if 'ref_channels' in opt_net.keys() else 3, nf=opt_net['nf'], scale=opt_net['scale'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD

# Discriminator
def define_D(opt, wrap=False):
    img_sz = opt['datasets']['train']['target_size']
    opt_net = opt['network_D']
    return define_D_net(opt_net, img_sz, wrap=wrap)

def define_fixed_D(opt):
    # Note that this will not work with "old" VGG-style discriminators with dense blocks until the img_size parameter is added.
    net = define_D_net(opt)

    # Load the model parameters:
    load_net = torch.load(opt['pretrained_path'])
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    net.load_state_dict(load_net_clean)

    # Put into eval mode, freeze the parameters and set the 'weight' field.
    net.eval()
    for k, v in net.named_parameters():
        v.requires_grad = False
    net.fdisc_weight = opt['weight']

    return net


# Define network used for perceptual loss
def define_F(which_model='vgg', use_bn=False, for_training=False, load_path=None):
    if which_model == 'vgg':
        # PyTorch pretrained VGG19-54, before ReLU.
        if use_bn:
            feature_layer = 49
        else:
            feature_layer = 34
        if for_training:
            netF = feature_arch.TrainableVGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                                  use_input_norm=True)
        else:
            netF = feature_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                                    use_input_norm=True)
    elif which_model == 'wide_resnet':
        netF = feature_arch.WideResnetFeatureExtractor(use_input_norm=True)
    else:
        raise NotImplementedError

    if load_path:
        # Load the model parameters:
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        netF.load_state_dict(load_net_clean)

    if not for_training:
        # Put into eval mode, freeze the parameters and set the 'weight' field.
        netF.eval()
        for k, v in netF.named_parameters():
            v.requires_grad = False

    return netF
