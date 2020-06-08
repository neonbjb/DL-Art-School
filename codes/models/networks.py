import torch
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.DiscriminatorResnet_arch as DiscriminatorResnet_arch
import models.archs.DiscriminatorResnet_arch_passthrough as DiscriminatorResnet_arch_passthrough
import models.archs.FlatProcessorNetNew_arch as FlatProcessorNetNew_arch
import models.archs.RRDBNet_arch as RRDBNet_arch
import models.archs.HighToLowResNet as HighToLowResNet
import models.archs.ResGen_arch as ResGen_arch
import models.archs.biggan_gen_arch as biggan_arch
import models.archs.feature_arch as feature_arch
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
    elif which_model == 'AttentiveRRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], scale=scale,
                                    rrdb_block_f=functools.partial(RRDBNet_arch.AttentiveRRDB, nf=opt_net['nf'], gc=opt_net['gc'],
                                                                   init_temperature=opt_net['temperature']))
    elif which_model == 'ResGen':
        netG = ResGen_arch.fixup_resnet34(nb_denoiser=opt_net['nb_denoiser'], nb_upsampler=opt_net['nb_upsampler'],
                                          upscale_applications=opt_net['upscale_applications'], num_filters=opt_net['nf'])
    elif which_model == 'ResGenV2':
        netG = ResGen_arch.fixup_resnet34_v2(nb_denoiser=opt_net['nb_denoiser'], nb_upsampler=opt_net['nb_upsampler'],
                                          upscale_applications=opt_net['upscale_applications'], num_filters=opt_net['nf'],
                                          inject_noise=opt_net['inject_noise'])
    elif which_model == "BigGan":
        netG = biggan_arch.biggan_medium(num_filters=opt_net['nf'])

    # image corruption
    elif which_model == 'HighToLowResNet':
        netG = HighToLowResNet.HighToLowResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                nf=opt_net['nf'], nb=opt_net['nb'], downscale=opt_net['scale'])
    elif which_model == 'FlatProcessorNet':
        '''netG = FlatProcessorNet_arch.FlatProcessorNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                nf=opt_net['nf'], downscale=opt_net['scale'], reduce_anneal_blocks=opt_net['ra_blocks'],
                                assembler_blocks=opt_net['assembler_blocks'])'''
        netG = FlatProcessorNetNew_arch.fixup_resnet34(num_filters=opt_net['nf'])
    # video restoration
    elif which_model == 'EDVR':
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'])

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    img_sz = opt['datasets']['train']['target_size']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'], input_img_factor=img_sz / 128)
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
