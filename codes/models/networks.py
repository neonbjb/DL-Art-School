import munch
import torch
import logging
from munch import munchify
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.DiscriminatorResnet_arch as DiscriminatorResnet_arch
import models.archs.DiscriminatorResnet_arch_passthrough as DiscriminatorResnet_arch_passthrough
import models.archs.RRDBNet_arch as RRDBNet_arch
import models.archs.feature_arch as feature_arch
import models.archs.SwitchedResidualGenerator_arch as SwitchedGen_arch
import models.archs.SPSR_arch as spsr
import models.archs.StructuredSwitchedGenerator as ssg
import models.archs.rcan as rcan
import models.archs.panet.panet as panet
from collections import OrderedDict
import torchvision
import functools

from models.archs.ChainedEmbeddingGen import ChainedEmbeddingGen, ChainedEmbeddingGenWithStructure, \
    ChainedEmbeddingGenWithStructureR2

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
                                    nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['scale'] if 'scale' in opt_net.keys() else gen_scale,
                                    initial_stride=initial_stride)
    elif which_model == 'rcan':
        #args: n_resgroups, n_resblocks, res_scale, reduction, scale, n_feats
        opt_net['rgb_range'] = 255
        opt_net['n_colors'] = 3
        args_obj = munchify(opt_net)
        netG = rcan.RCAN(args_obj)
    elif which_model == 'panet':
        #args: n_resblocks, res_scale, scale, n_feats
        opt_net['rgb_range'] = 255
        opt_net['n_colors'] = 3
        args_obj = munchify(opt_net)
        netG = panet.PANET(args_obj)
    elif which_model == "ConfigurableSwitchedResidualGenerator2":
        netG = SwitchedGen_arch.ConfigurableSwitchedResidualGenerator2(switch_depth=opt_net['switch_depth'], switch_filters=opt_net['switch_filters'],
                                                                      switch_reductions=opt_net['switch_reductions'],
                                                                      switch_processing_layers=opt_net['switch_processing_layers'], trans_counts=opt_net['trans_counts'],
                                                                      trans_kernel_sizes=opt_net['trans_kernel_sizes'], trans_layers=opt_net['trans_layers'],
                                                                      transformation_filters=opt_net['transformation_filters'], attention_norm=opt_net['attention_norm'],
                                                                      initial_temp=opt_net['temperature'], final_temperature_step=opt_net['temperature_final_step'],
                                                                      heightened_temp_min=opt_net['heightened_temp_min'], heightened_final_step=opt_net['heightened_final_step'],
                                                                      upsample_factor=scale, add_scalable_noise_to_transforms=opt_net['add_noise'])
    elif which_model == 'spsr_net_improved':
        netG = spsr.SPSRNetSimplified(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                            nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == "spsr_switched":
        netG = spsr.SwitchedSpsr(in_nc=3, nf=opt_net['nf'], upscale=opt_net['scale'], init_temperature=opt_net['temperature'])
    elif which_model == "spsr5":
        xforms = opt_net['num_transforms'] if 'num_transforms' in opt_net.keys() else 8
        netG = spsr.Spsr5(in_nc=3, out_nc=3, nf=opt_net['nf'], xforms=xforms, upscale=opt_net['scale'],
                                 multiplexer_reductions=opt_net['multiplexer_reductions'] if 'multiplexer_reductions' in opt_net.keys() else 2,
                                 init_temperature=opt_net['temperature'] if 'temperature' in opt_net.keys() else 10)
    elif which_model == "spsr6":
        xforms = opt_net['num_transforms'] if 'num_transforms' in opt_net.keys() else 8
        netG = spsr.Spsr6(in_nc=3, out_nc=3, nf=opt_net['nf'], xforms=xforms, upscale=opt_net['scale'],
                                 multiplexer_reductions=opt_net['multiplexer_reductions'] if 'multiplexer_reductions' in opt_net.keys() else 3,
                                 init_temperature=opt_net['temperature'] if 'temperature' in opt_net.keys() else 10)
    elif which_model == "spsr7":
        recurrent = opt_net['recurrent'] if 'recurrent' in opt_net.keys() else False
        xforms = opt_net['num_transforms'] if 'num_transforms' in opt_net.keys() else 8
        netG = spsr.Spsr7(in_nc=3, out_nc=3, nf=opt_net['nf'], xforms=xforms, upscale=opt_net['scale'],
                                 multiplexer_reductions=opt_net['multiplexer_reductions'] if 'multiplexer_reductions' in opt_net.keys() else 3,
                                 init_temperature=opt_net['temperature'] if 'temperature' in opt_net.keys() else 10, recurrent=recurrent)
    elif which_model == "spsr9":
        xforms = opt_net['num_transforms'] if 'num_transforms' in opt_net.keys() else 8
        netG = spsr.Spsr9(in_nc=3, out_nc=3, nf=opt_net['nf'], xforms=xforms, upscale=opt_net['scale'],
                                 multiplexer_reductions=opt_net['multiplexer_reductions'] if 'multiplexer_reductions' in opt_net.keys() else 3,
                                 init_temperature=opt_net['temperature'] if 'temperature' in opt_net.keys() else 10)
    elif which_model == "ssgr1":
        recurrent = opt_net['recurrent'] if 'recurrent' in opt_net.keys() else False
        xforms = opt_net['num_transforms'] if 'num_transforms' in opt_net.keys() else 8
        netG = ssg.SSGr1(in_nc=3, out_nc=3, nf=opt_net['nf'], xforms=xforms, upscale=opt_net['scale'],
                                 init_temperature=opt_net['temperature'] if 'temperature' in opt_net.keys() else 10, recurrent=recurrent)
    elif which_model == 'stacked_switches':
        xforms = opt_net['num_transforms'] if 'num_transforms' in opt_net.keys() else 8
        in_nc = opt_net['in_nc'] if 'in_nc' in opt_net.keys() else 3
        netG = ssg.StackedSwitchGenerator(in_nc=in_nc, out_nc=3, nf=opt_net['nf'], xforms=xforms, upscale=opt_net['scale'],
                                 init_temperature=opt_net['temperature'] if 'temperature' in opt_net.keys() else 10)
    elif which_model == 'stacked_switches_5lyr':
        xforms = opt_net['num_transforms'] if 'num_transforms' in opt_net.keys() else 8
        in_nc = opt_net['in_nc'] if 'in_nc' in opt_net.keys() else 3
        netG = ssg.StackedSwitchGenerator5Layer(in_nc=in_nc, out_nc=3, nf=opt_net['nf'], xforms=xforms, upscale=opt_net['scale'],
                                 init_temperature=opt_net['temperature'] if 'temperature' in opt_net.keys() else 10)
    elif which_model == 'ssg_deep':
        xforms = opt_net['num_transforms'] if 'num_transforms' in opt_net.keys() else 8
        netG = ssg.SSGDeep(in_nc=3, out_nc=3, nf=opt_net['nf'], xforms=xforms, upscale=opt_net['scale'],
                                 init_temperature=opt_net['temperature'] if 'temperature' in opt_net.keys() else 10)
    elif which_model == 'ssg_simpler':
        xforms = opt_net['num_transforms'] if 'num_transforms' in opt_net.keys() else 8
        netG = ssg.SsgSimpler(in_nc=3, out_nc=3, nf=opt_net['nf'], xforms=xforms,
                                 init_temperature=opt_net['temperature'] if 'temperature' in opt_net.keys() else 10)
    elif which_model == 'ssg_teco':
        netG = ssg.StackedSwitchGenerator2xTeco(nf=opt_net['nf'], xforms=opt_net['num_transforms'], init_temperature=opt_net['temperature'] if 'temperature' in opt_net.keys() else 10)
    elif which_model == 'big_switch':
        netG = SwitchedGen_arch.TheBigSwitch(opt_net['in_nc'], nf=opt_net['nf'], xforms=opt_net['num_transforms'], upscale=opt_net['scale'],
                                             init_temperature=opt_net['temperature'])
    elif which_model == 'artist':
        netG = SwitchedGen_arch.ArtistGen(opt_net['in_nc'], nf=opt_net['nf'], xforms=opt_net['num_transforms'], upscale=opt_net['scale'],
                                             init_temperature=opt_net['temperature'])
    elif which_model == 'chained_gen':
        netG = ChainedEmbeddingGen(depth=opt_net['depth'])
    elif which_model == 'chained_gen_structured':
        netG = ChainedEmbeddingGenWithStructure(depth=opt_net['depth'])
    elif which_model == 'chained_gen_structuredr2':
        netG = ChainedEmbeddingGenWithStructureR2(depth=opt_net['depth'])
    elif which_model == "flownet2":
        from models.flownet2.models import FlowNet2
        ld = torch.load(opt_net['load_path'])
        args = munch.Munch({'fp16': False, 'rgb_max': 1.0})
        netG = FlowNet2(args)
        netG.load_state_dict(ld['state_dict'])
    elif which_model == "backbone_encoder":
        netG = SwitchedGen_arch.BackboneEncoder(pretrained_backbone=opt_net['pretrained_spinenet'])
    elif which_model == "backbone_encoder_no_ref":
        netG = SwitchedGen_arch.BackboneEncoderNoRef(pretrained_backbone=opt_net['pretrained_spinenet'])
    elif which_model == "backbone_encoder_no_head":
        netG = SwitchedGen_arch.BackboneSpinenetNoHead()
    elif which_model == "backbone_resnet":
        netG = SwitchedGen_arch.BackboneResnet()
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

    if 'image_size' in opt_net.keys():
        img_sz = opt_net['image_size']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'], input_img_factor=img_sz / 128, extra_conv=opt_net['extra_conv'])
    elif which_model == 'discriminator_vgg_128_gn':
        netD = SRGAN_arch.Discriminator_VGG_128_GN(in_nc=opt_net['in_nc'], nf=opt_net['nf'], input_img_factor=img_sz / 128)
        if wrap:
            netD = GradDiscWrapper(netD)
    elif which_model == 'discriminator_resnet':
        netD = DiscriminatorResnet_arch.fixup_resnet34(num_filters=opt_net['nf'], num_classes=1, input_img_size=img_sz)
    elif which_model == 'discriminator_resnet_50':
        netD = DiscriminatorResnet_arch.fixup_resnet50(num_filters=opt_net['nf'], num_classes=1, input_img_size=img_sz)
    elif which_model == 'discriminator_resnet_passthrough':
        netD = DiscriminatorResnet_arch_passthrough.fixup_resnet34(num_filters=opt_net['nf'], num_classes=1, input_img_size=img_sz,
                                                                   number_skips=opt_net['number_skips'], use_bn=True,
                                                                   disable_passthrough=opt_net['disable_passthrough'])
    elif which_model == 'resnext':
        netD = torchvision.models.resnext50_32x4d(norm_layer=functools.partial(torch.nn.GroupNorm, 8))
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth', progress=True)
        netD.load_state_dict(state_dict, strict=False)
        netD.fc = torch.nn.Linear(512 * 4, 1)
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
    elif which_model == "discriminator_refvgg":
        netD = SRGAN_arch.RefDiscriminatorVgg128(in_nc=opt_net['in_nc'], nf=opt_net['nf'], input_img_factor=img_sz / 128)
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
def define_F(which_model='vgg', use_bn=False, for_training=False, load_path=None, feature_layers=None):
    if which_model == 'vgg':
        # PyTorch pretrained VGG19-54, before ReLU.
        if feature_layers is None:
            if use_bn:
                feature_layers = [49]
            else:
                feature_layers = [34]
        if for_training:
            netF = feature_arch.TrainableVGGFeatureExtractor(feature_layers=feature_layers, use_bn=use_bn,
                                                  use_input_norm=True)
        else:
            netF = feature_arch.VGGFeatureExtractor(feature_layers=feature_layers, use_bn=use_bn,
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
