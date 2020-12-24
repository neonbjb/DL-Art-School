import functools
import logging
from collections import OrderedDict

import munch
import torch
import torchvision
from munch import munchify
import models.stylegan.stylegan2_lucidrains as stylegan2

import models.fixup_resnet.DiscriminatorResnet_arch as DiscriminatorResnet_arch
import models.RRDBNet_arch as RRDBNet_arch
import models.SwitchedResidualGenerator_arch as SwitchedGen_arch
import models.discriminator_vgg_arch as SRGAN_arch
import models.feature_arch as feature_arch
from models import srg2_classic
from models.stylegan.Discriminator_StyleGAN import StyleGanDiscriminator
from models.tecogan.teco_resgen import TecoGen
from utils.util import opt_get

logger = logging.getLogger('base')

# Generator
def define_G(opt, opt_net, scale=None):
    if scale is None:
        scale = opt['scale']
    which_model = opt_net['which_model_G']

    if 'RRDBNet' in which_model:
        if which_model == 'RRDBNetBypass':
            block = RRDBNet_arch.RRDBWithBypass
        elif which_model == 'RRDBNetLambda':
            from models.lambda_rrdb import LambdaRRDB
            block = LambdaRRDB
        else:
            block = RRDBNet_arch.RRDB
        additive_mode = opt_net['additive_mode'] if 'additive_mode' in opt_net.keys() else 'not'
        output_mode = opt_net['output_mode'] if 'output_mode' in opt_net.keys() else 'hq_only'
        gc = opt_net['gc'] if 'gc' in opt_net.keys() else 32
        initial_stride = opt_net['initial_stride'] if 'initial_stride' in opt_net.keys() else 1
        netG = RRDBNet_arch.RRDBNet(in_channels=opt_net['in_nc'], out_channels=opt_net['out_nc'],
                                    mid_channels=opt_net['nf'], num_blocks=opt_net['nb'], additive_mode=additive_mode,
                                    output_mode=output_mode, body_block=block, scale=opt_net['scale'], growth_channels=gc,
                                    initial_stride=initial_stride)
    elif which_model == "ConfigurableSwitchedResidualGenerator2":
        netG = SwitchedGen_arch.ConfigurableSwitchedResidualGenerator2(switch_depth=opt_net['switch_depth'], switch_filters=opt_net['switch_filters'],
                                                                      switch_reductions=opt_net['switch_reductions'],
                                                                      switch_processing_layers=opt_net['switch_processing_layers'], trans_counts=opt_net['trans_counts'],
                                                                      trans_kernel_sizes=opt_net['trans_kernel_sizes'], trans_layers=opt_net['trans_layers'],
                                                                      transformation_filters=opt_net['transformation_filters'], attention_norm=opt_net['attention_norm'],
                                                                      initial_temp=opt_net['temperature'], final_temperature_step=opt_net['temperature_final_step'],
                                                                      heightened_temp_min=opt_net['heightened_temp_min'], heightened_final_step=opt_net['heightened_final_step'],
                                                                      upsample_factor=scale, add_scalable_noise_to_transforms=opt_net['add_noise'],
                                                                      for_video=opt_net['for_video'])
    elif which_model == "srg2classic":
        netG = srg2_classic.ConfigurableSwitchedResidualGenerator2(switch_depth=opt_net['switch_depth'], switch_filters=opt_net['switch_filters'],
                                                                      switch_reductions=opt_net['switch_reductions'],
                                                                      switch_processing_layers=opt_net['switch_processing_layers'], trans_counts=opt_net['trans_counts'],
                                                                      trans_kernel_sizes=opt_net['trans_kernel_sizes'], trans_layers=opt_net['trans_layers'],
                                                                      transformation_filters=opt_net['transformation_filters'],
                                                                      initial_temp=opt_net['temperature'], final_temperature_step=opt_net['temperature_final_step'],
                                                                      heightened_temp_min=opt_net['heightened_temp_min'], heightened_final_step=opt_net['heightened_final_step'],
                                                                      upsample_factor=scale, add_scalable_noise_to_transforms=opt_net['add_noise'])
    elif which_model == "flownet2":
        from models.flownet2 import FlowNet2
        ld = 'load_path' in opt_net.keys()
        args = munch.Munch({'fp16': False, 'rgb_max': 1.0, 'checkpoint': not ld})
        netG = FlowNet2(args)
        if ld:
            sd = torch.load(opt_net['load_path'])
            netG.load_state_dict(sd['state_dict'])
    elif which_model == "backbone_encoder":
        netG = SwitchedGen_arch.BackboneEncoder(pretrained_backbone=opt_net['pretrained_spinenet'])
    elif which_model == "backbone_encoder_no_ref":
        netG = SwitchedGen_arch.BackboneEncoderNoRef(pretrained_backbone=opt_net['pretrained_spinenet'])
    elif which_model == "backbone_encoder_no_head":
        netG = SwitchedGen_arch.BackboneSpinenetNoHead()
    elif which_model == "backbone_resnet":
        netG = SwitchedGen_arch.BackboneResnet()
    elif which_model == "tecogen":
        netG = TecoGen(opt_net['nf'], opt_net['scale'])
    elif which_model == 'stylegan2':
        is_structured = opt_net['structured'] if 'structured' in opt_net.keys() else False
        attn = opt_net['attn_layers'] if 'attn_layers' in opt_net.keys() else []
        netG = stylegan2.StyleGan2GeneratorWithLatent(image_size=opt_net['image_size'], latent_dim=opt_net['latent_dim'],
                                            style_depth=opt_net['style_depth'], structure_input=is_structured,
                                            attn_layers=attn)
    elif which_model == 'srflow':
        from models.srflow import SRFlowNet_arch
        netG = SRFlowNet_arch.SRFlowNet(in_nc=3, out_nc=3, nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['scale'],
                                     K=opt_net['K'], opt=opt)
    elif which_model == 'rrdb_latent_wrapper':
        from models.srflow.RRDBNet_arch import RRDBLatentWrapper
        netG = RRDBLatentWrapper(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                  nf=opt_net['nf'], nb=opt_net['nb'], with_bypass=opt_net['with_bypass'],
                                 blocks=opt_net['blocks_for_latent'], scale=opt_net['scale'], pretrain_rrdb_path=opt_net['pretrain_path'])
    elif which_model == 'rrdb_centipede':
        output_mode = opt_net['output_mode'] if 'output_mode' in opt_net.keys() else 'hq_only'
        netG = RRDBNet_arch.RRDBNet(in_channels=opt_net['in_nc'], out_channels=opt_net['out_nc'],
                                    mid_channels=opt_net['nf'], num_blocks=opt_net['nb'], scale=opt_net['scale'],
                                    headless=True, output_mode=output_mode)
    elif which_model == 'rrdb_srflow':
        from models.srflow.RRDBNet_arch import RRDBNet
        netG = RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                       nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['scale'],
                       initial_conv_stride=opt_net['initial_stride'])
    elif which_model == 'igpt2':
        from models.transformers.igpt.gpt2 import iGPT2
        netG = iGPT2(opt_net['embed_dim'], opt_net['num_heads'], opt_net['num_layers'], opt_net['num_pixels'] ** 2, opt_net['num_vocab'], centroids_file=opt_net['centroids_file'])
    elif which_model == 'byol':
        from models.byol.byol_model_wrapper import BYOL
        subnet = define_G(opt, opt_net['subnet'])
        netG = BYOL(subnet, opt_net['image_size'], opt_net['hidden_layer'],
                    structural_mlp=opt_get(opt_net, ['use_structural_mlp'], False))
    elif which_model == 'structural_byol':
        from models.byol.byol_structural import StructuralBYOL
        subnet = define_G(opt, opt_net['subnet'])
        netG = StructuralBYOL(subnet, opt_net['image_size'], opt_net['hidden_layer'],
                              pretrained_state_dict=opt_get(opt_net, ["pretrained_path"]),
                              freeze_until=opt_get(opt_net, ['freeze_until'], 0))
    elif which_model == 'spinenet':
        from models.spinenet_arch import SpineNet
        netG = SpineNet(str(opt_net['arch']), in_channels=3, use_input_norm=opt_net['use_input_norm'])
    elif which_model == 'spinenet_with_logits':
        from models.spinenet_arch import SpinenetWithLogits
        netG = SpinenetWithLogits(str(opt_net['arch']), opt_net['output_to_attach'], opt_net['num_labels'],
                        in_channels=3, use_input_norm=opt_net['use_input_norm'])
    elif which_model == 'resnet52':
        from models.resnet_with_checkpointing import resnet50
        netG = resnet50(pretrained=opt_net['pretrained'])
    elif which_model == 'glean':
        from models.glean.glean import GleanGenerator
        netG = GleanGenerator(opt_net['nf'], opt_net['pretrained_stylegan'])
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
        extra_conv = opt_net['extra_conv'] if 'extra_conv' in opt_net.keys() else False
        netD = SRGAN_arch.Discriminator_VGG_128_GN(in_nc=opt_net['in_nc'], nf=opt_net['nf'],
                                                   input_img_factor=img_sz / 128, extra_conv=extra_conv)
        if wrap:
            netD = GradDiscWrapper(netD)
    elif which_model == 'discriminator_vgg_128_gn_checkpointed':
        netD = SRGAN_arch.Discriminator_VGG_128_GN(in_nc=opt_net['in_nc'], nf=opt_net['nf'], input_img_factor=img_sz / 128, do_checkpointing=True)
    elif which_model == 'stylegan_vgg':
        netD = StyleGanDiscriminator(128)
    elif which_model == 'discriminator_resnet':
        netD = DiscriminatorResnet_arch.fixup_resnet34(num_filters=opt_net['nf'], num_classes=1, input_img_size=img_sz)
    elif which_model == 'discriminator_resnet_50':
        netD = DiscriminatorResnet_arch.fixup_resnet50(num_filters=opt_net['nf'], num_classes=1, input_img_size=img_sz)
    elif which_model == 'resnext':
        netD = torchvision.models.resnext50_32x4d(norm_layer=functools.partial(torch.nn.GroupNorm, 8))
        #state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth', progress=True)
        #netD.load_state_dict(state_dict, strict=False)
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
    elif which_model == "psnr_approximator":
        netD = SRGAN_arch.PsnrApproximator(nf=opt_net['nf'], input_img_factor=img_sz / 128)
    elif which_model == "stylegan2_discriminator":
        attn = opt_net['attn_layers'] if 'attn_layers' in opt_net.keys() else []
        disc = stylegan2.StyleGan2Discriminator(image_size=opt_net['image_size'], input_filters=opt_net['in_nc'], attn_layers=attn)
        netD = stylegan2.StyleGan2Augmentor(disc, opt_net['image_size'], types=opt_net['augmentation_types'], prob=opt_net['augmentation_probability'])
    elif which_model == "rrdb_disc":
        netD = RRDBNet_arch.RRDBDiscriminator(opt_net['in_nc'], opt_net['nf'], opt_net['nb'], blocks_per_checkpoint=3)
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
