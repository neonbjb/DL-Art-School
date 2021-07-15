import importlib
import logging
import os
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
        if os.getcwd() not in mod.module_finder.path:
            continue   # I have no idea why this is necessary - I think it's a bug in the latest PyWindows release.
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