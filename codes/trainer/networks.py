import importlib
import logging
import pkgutil
import sys
from collections import OrderedDict
from inspect import isfunction, getmembers, signature
import torch
import models.feature_arch as feature_arch

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


def create_model(opt, opt_net, other_nets=None):
    which_model = opt_net['which_model']
    # For backwards compatibility.
    if not which_model:
        which_model = opt_net['which_model_G']
    if not which_model:
        which_model = opt_net['which_model_D']
    registered_fns = find_registered_model_fns()
    if which_model not in registered_fns.keys():
        raise CreateModelError(which_model, list(registered_fns.keys()))
    num_params = len(signature(registered_fns[which_model]).parameters)
    if num_params == 2:
        return registered_fns[which_model](opt_net, opt)
    else:
        return registered_fns[which_model](opt_net, opt, other_nets)


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
