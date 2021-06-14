# Base class for an evaluator, which is responsible for feeding test data through a model and evaluating the response.
import importlib
import inspect
import pkgutil
import re
import sys


class Evaluator:
    def __init__(self, model, opt_eval, env, uses_all_ddp=True):
        self.model = model.module if hasattr(model, 'module') else model
        self.opt = opt_eval
        self.env = env
        self.uses_all_ddp = uses_all_ddp

    def perform_eval(self):
        return {}


def format_evaluator_name(name):
    # Formats by converting from CamelCase to snake_case and removing trailing "_evaluator"
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    return name.replace("_evaluator", "")


# Works by loading all python modules in the eval/ directory and sniffing out subclasses of Evaluator.
def find_registered_evaluators(base_path="trainer/eval"):
    module_iter = pkgutil.walk_packages([base_path])
    results = {}
    for mod in module_iter:
        if mod.ispkg:
            EXCLUSION_LIST = []
            if mod.name not in EXCLUSION_LIST:
                results.update(find_registered_evaluators(f'{base_path}/{mod.name}'))
        else:
            mod_name = f'{base_path}/{mod.name}'.replace('/', '.')
            importlib.import_module(mod_name)
            classes = inspect.getmembers(sys.modules[mod_name], inspect.isclass)
            for name, obj in classes:
                if 'Evaluator' in [mro.__name__ for mro in inspect.getmro(obj)]:
                    results[format_evaluator_name(name)] = obj
    return results


class CreateEvaluatorError(Exception):
    def __init__(self, name, available):
        super().__init__(f'Could not find the specified evaluator name: {name}.  Available evaluators:'
                         f'{available}')


def create_evaluator(model, opt_eval, env):
    evaluators = find_registered_evaluators()
    type = opt_eval['type']
    if type not in evaluators.keys():
        raise CreateEvaluatorError(type, list(evaluators.keys()))
    return evaluators[opt_eval['type']](model, opt_eval, env)
