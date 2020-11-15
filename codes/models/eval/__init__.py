from models.eval.sr_style import SrStyleTransferEvaluator
from models.eval.style import StyleTransferEvaluator


def create_evaluator(model, opt_eval, env):
    type = opt_eval['type']
    if type == 'style_transfer':
        return StyleTransferEvaluator(model, opt_eval, env)
    elif type == 'sr_stylegan':
        return SrStyleTransferEvaluator(model, opt_eval, env)
    else:
        raise NotImplementedError()