from trainer.eval.flow_gaussian_nll import FlowGaussianNll
from trainer.eval.sr_style import SrStyleTransferEvaluator
from trainer.eval import StyleTransferEvaluator


def create_evaluator(model, opt_eval, env):
    type = opt_eval['type']
    if type == 'style_transfer':
        return StyleTransferEvaluator(model, opt_eval, env)
    elif type == 'sr_stylegan':
        return SrStyleTransferEvaluator(model, opt_eval, env)
    elif type == 'flownet_gaussian':
        return FlowGaussianNll(model, opt_eval, env)
    else:
        raise NotImplementedError()