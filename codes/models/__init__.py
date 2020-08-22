import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration
    if model == 'sr':  # PSNR-oriented super resolution
        from .SR_model import SRModel as M
    elif model == 'srgan' or model == 'corruptgan' or model == 'spsrgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'feat':
        from .feature_model import FeatureModel as M
    elif model == 'spsr':
        from .SPSR_model import SPSRModel as M
    elif model == 'extensibletrainer':
        from .ExtensibleTrainer import ExtensibleTrainer as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
