import importlib
import inspect
import pkgutil
import re
import sys

import torch.nn


# Base class for all other injectors.
class Injector(torch.nn.Module):
    def __init__(self, opt, env):
        super(Injector, self).__init__()
        self.opt = opt
        self.env = env
        if 'in' in opt.keys():
            self.input = opt['in']
        if 'out' in opt.keys():
            self.output = opt['out']

    # This should return a dict of new state variables.
    def forward(self, state):
        raise NotImplementedError


def format_injector_name(name):
    # Formats by converting from CamelCase to snake_case and removing trailing "_injector"
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    return name.replace("_injector", "")


# Works by loading all python modules in the injectors/ directory and sniffing out subclasses of Injector.
# field will be properly populated.
def find_registered_injectors(base_path="trainer/injectors"):
    module_iter = pkgutil.walk_packages([base_path])
    results = {}
    for mod in module_iter:
        if mod.ispkg:
            EXCLUSION_LIST = []
            if mod.name not in EXCLUSION_LIST:
                results.update(find_registered_injectors(f'{base_path}/{mod.name}'))
        else:
            mod_name = f'{base_path}/{mod.name}'.replace('/', '.')
            importlib.import_module(mod_name)
            classes = inspect.getmembers(sys.modules[mod_name], inspect.isclass)
            for name, obj in classes:
                if 'Injector' in [mro.__name__ for mro in inspect.getmro(obj)]:
                    results[format_injector_name(name)] = obj
    return results


class CreateInjectorError(Exception):
    def __init__(self, name, available):
        super().__init__(f'Could not find the specified injector name: {name}.  Available injectors:'
                         f'{available}')


# Injectors are a way to synthesize data within a step that can then be used (and reused) by loss functions.
def create_injector(opt_inject, env):
    injectors = find_registered_injectors()
    type = opt_inject['type']
    if type not in injectors.keys():
        raise CreateInjectorError(type, list(injectors.keys()))
    return injectors[opt_inject['type']](opt_inject, env)
