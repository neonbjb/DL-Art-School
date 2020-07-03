import torch
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
import functools

# Generator
def define_G(opt, net_key='network_G'):
    opt_net = opt[net_key]
    which_model = opt_net['which_model_G']
    scale = opt['scale']

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
    elif which_model == 'AssistedRRDBNet':
        netG = RRDBNet_arch.AssistedRRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], scale=scale)
    elif which_model == 'LowDimRRDBNet':
        gen_scale = scale * opt_net['initial_stride']
        rrdb = functools.partial(RRDBNet_arch.LowDimRRDB, nf=opt_net['nf'], gc=opt_net['gc'], dimensional_adjustment=opt_net['dim'])
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], scale=gen_scale, rrdb_block_f=rrdb, initial_stride=opt_net['initial_stride'])
    elif which_model == 'PixRRDBNet':
        block_f = None
        if opt_net['attention']:
            block_f = functools.partial(RRDBNet_arch.SwitchedRRDB, nf=opt_net['nf'], gc=opt_net['gc'],
                                        init_temperature=opt_net['temperature'],
                                        final_temperature_step=opt_net['temperature_final_step'])
        if opt_net['mhattention']:
            block_f = functools.partial(RRDBNet_arch.SwitchedMultiHeadRRDB, num_convs=8, num_heads=2, nf=opt_net['nf'], gc=opt_net['gc'],
                                        init_temperature=opt_net['temperature'],
                                        final_temperature_step=opt_net['temperature_final_step'])
        netG = RRDBNet_arch.PixShuffleRRDB(nf=opt_net['nf'], nb=opt_net['nb'], gc=opt_net['gc'], scale=scale, rrdb_block_f=block_f)
    elif which_model == "ConfigurableSwitchedResidualGenerator":
        netG = SwitchedGen_arch.ConfigurableSwitchedResidualGenerator(switch_filters=opt_net['switch_filters'], switch_growths=opt_net['switch_growths'],
                                                                      switch_reductions=opt_net['switch_reductions'],
                                                                      switch_processing_layers=opt_net['switch_processing_layers'], trans_counts=opt_net['trans_counts'],
                                                                      trans_kernel_sizes=opt_net['trans_kernel_sizes'], trans_layers=opt_net['trans_layers'],
                                                                      trans_filters_mid=opt_net['trans_filters_mid'],
                                                                      initial_temp=opt_net['temperature'], final_temperature_step=opt_net['temperature_final_step'],
                                                                      heightened_temp_min=opt_net['heightened_temp_min'], heightened_final_step=opt_net['heightened_final_step'],
                                                                      upsample_factor=scale, add_scalable_noise_to_transforms=opt_net['add_noise'])
    elif which_model == "ConfigurableSwitchedResidualGenerator2":
        netG = SwitchedGen_arch.ConfigurableSwitchedResidualGenerator2(switch_filters=opt_net['switch_filters'], switch_growths=opt_net['switch_growths'],
                                                                      switch_reductions=opt_net['switch_reductions'],
                                                                      switch_processing_layers=opt_net['switch_processing_layers'], trans_counts=opt_net['trans_counts'],
                                                                      trans_kernel_sizes=opt_net['trans_kernel_sizes'], trans_layers=opt_net['trans_layers'],
                                                                      transformation_filters=opt_net['transformation_filters'],
                                                                      initial_temp=opt_net['temperature'], final_temperature_step=opt_net['temperature_final_step'],
                                                                      heightened_temp_min=opt_net['heightened_temp_min'], heightened_final_step=opt_net['heightened_final_step'],
                                                                      upsample_factor=scale, add_scalable_noise_to_transforms=opt_net['add_noise'])
    elif which_model == "ConfigurableSwitchedResidualGenerator3":
        netG = SwitchedGen_arch.ConfigurableSwitchedResidualGenerator3(trans_counts=opt_net['trans_counts'],
                                                                      trans_kernel_sizes=opt_net['trans_kernel_sizes'], trans_layers=opt_net['trans_layers'],
                                                                      transformation_filters=opt_net['transformation_filters'],
                                                                      initial_temp=opt_net['temperature'], final_temperature_step=opt_net['temperature_final_step'],
                                                                      heightened_temp_min=opt_net['heightened_temp_min'], heightened_final_step=opt_net['heightened_final_step'],
                                                                      upsample_factor=scale, add_scalable_noise_to_transforms=opt_net['add_noise'])
    elif which_model == "NestedSwitchGenerator":
        netG = ng.NestedSwitchedGenerator(switch_filters=opt_net['switch_filters'],
                                                                      switch_reductions=opt_net['switch_reductions'],
                                                                      switch_processing_layers=opt_net['switch_processing_layers'], trans_counts=opt_net['trans_counts'],
                                                                      trans_kernel_sizes=opt_net['trans_kernel_sizes'], trans_layers=opt_net['trans_layers'],
                                                                      transformation_filters=opt_net['transformation_filters'],
                                                                      initial_temp=opt_net['temperature'], final_temperature_step=opt_net['temperature_final_step'],
                                                                      heightened_temp_min=opt_net['heightened_temp_min'], heightened_final_step=opt_net['heightened_final_step'],
                                                                      upsample_factor=scale, add_scalable_noise_to_transforms=opt_net['add_noise'])

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


# Discriminator
def define_D(opt):
    img_sz = opt['datasets']['train']['target_size']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'], input_img_factor=img_sz // 128, extra_conv=opt_net['extra_conv'])
    elif which_model == 'discriminator_resnet':
        netD = DiscriminatorResnet_arch.fixup_resnet34(num_filters=opt_net['nf'], num_classes=1, input_img_size=img_sz)
    elif which_model == 'discriminator_resnet_passthrough':
        netD = DiscriminatorResnet_arch_passthrough.fixup_resnet34(num_filters=opt_net['nf'], num_classes=1, input_img_size=img_sz,
                                                                   number_skips=opt_net['number_skips'], use_bn=True,
                                                                   disable_passthrough=opt_net['disable_passthrough'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    if 'which_model_F' not in opt['train'].keys() or opt['train']['which_model_F'] == 'vgg':
        # PyTorch pretrained VGG19-54, before ReLU.
        if use_bn:
            feature_layer = 49
        else:
            feature_layer = 34
        netF = feature_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                              use_input_norm=True, device=device)
    elif opt['train']['which_model_F'] == 'wide_resnet':
        netF = feature_arch.WideResnetFeatureExtractor(use_input_norm=True, device=device)

    netF.eval()  # No need to train
    return netF
