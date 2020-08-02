import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration
    if model == 'sr':  # PSNR-oriented super resolution
        from .SR_model import SRModel as M
    elif model == 'srgan' or model == 'corruptgan':  # GAN-based super resolution(SRGAN / ESRGAN), or corruption use same logic
        from .SRGAN_model import SRGANModel as M
    elif model == 'feat':
        from .feature_model import FeatureModel as M
    if model == 'spsr':
        from .SPSR_model import SPSRModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
