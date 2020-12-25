import functools
import importlib
import logging
import pkgutil
import sys
from collections import OrderedDict
from inspect import isfunction, getmembers

import torch
import torchvision

import models.discriminator_vgg_arch as SRGAN_arch
import models.feature_arch as feature_arch
import models.fixup_resnet.DiscriminatorResnet_arch as DiscriminatorResnet_arch
from models.stylegan.Discriminator_StyleGAN import StyleGanDiscriminator

logger = logging.getLogger('base')


class RegisteredModelNameError(Exception):
    def __init__(self, name_error):
        super().__init__(f'Registered DLAS modules must start with `register_`. Incorrect registration: {name_error}')


# Decorator that allows API clients to show DLAS how to build a nn.Module from an opt dict.
# Functions with this decorator should have a specific naming format:
# `register_<name>` where <name> is the name that will be used in configuration files to reference this model.
# Functions with this decorator are expected to take a single argument:
# - opt: A dict with the configuration options for building the module.
# They should return:
# - A torch.nn.Module object for the model being defined.
def register_model(func):
    if func.__name__.startswith("register_"):
        func._dlas_model_name = func.__name__[9:]
        assert func._dlas_model_name
    else:
        raise RegisteredModelNameError(func.__name__)
    func._dlas_registered_model = True
    return func


def find_registered_model_fns(base_path='models'):
    found_fns = {}
    module_iter = pkgutil.walk_packages([base_path])
    for mod in module_iter:
        if mod.ispkg:
            EXCLUSION_LIST = ['flownet2']
            if mod.name not in EXCLUSION_LIST:
                found_fns.update(find_registered_model_fns(f'{base_path}/{mod.name}'))
        else:
            mod_name = f'{base_path}/{mod.name}'.replace('/', '.')
            importlib.import_module(mod_name)
            for mod_fn in getmembers(sys.modules[mod_name], isfunction):
                if hasattr(mod_fn[1], "_dlas_registered_model"):
                    found_fns[mod_fn[1]._dlas_model_name] = mod_fn[1]
    return found_fns


class CreateModelError(Exception):
    def __init__(self, name, available):
        super().__init__(f'Could not find the specified model name: {name}. Tip: If your model is in a'
                         f' subdirectory, that directory must contain an __init__.py to be scanned. Available models:'
                         f'{available}')


def create_model(opt, opt_net, scale=None):
    which_model = opt_net['which_model']
    # For backwards compatibility.
    if not which_model:
        which_model = opt_net['which_model_G']
    if not which_model:
        which_model = opt_net['which_model_D']
    registered_fns = find_registered_model_fns()
    if which_model not in registered_fns.keys():
        raise CreateModelError(which_model, list(registered_fns.keys()))
    return registered_fns[which_model](opt_net, opt)


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
