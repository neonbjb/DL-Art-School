from models.archs.stylegan.stylegan2 import StyleGan2DivergenceLoss, StyleGan2PathLengthLoss
from models.archs.stylegan.stylegan2_unet_disc import StyleGan2UnetDivergenceLoss


def create_stylegan2_loss(opt_loss, env):
    type = opt_loss['type']
    if type == 'stylegan2_divergence':
        return StyleGan2DivergenceLoss(opt_loss, env)
    elif type == 'stylegan2_pathlen':
        return StyleGan2PathLengthLoss(opt_loss, env)
    elif type == 'stylegan2_unet_divergence':
        return StyleGan2UnetDivergenceLoss(opt_loss, env)
    else:
        raise NotImplementedError