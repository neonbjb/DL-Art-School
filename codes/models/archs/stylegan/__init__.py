import models.archs.stylegan.stylegan2 as stylegan2
import models.archs.stylegan.stylegan2_unet_disc as stylegan2_unet


def create_stylegan2_loss(opt_loss, env):
    type = opt_loss['type']
    if type == 'stylegan2_divergence':
        return stylegan2.StyleGan2DivergenceLoss(opt_loss, env)
    elif type == 'stylegan2_pathlen':
        return stylegan2.StyleGan2PathLengthLoss(opt_loss, env)
    elif type == 'stylegan2_unet_divergence':
        return stylegan2_unet.StyleGan2UnetDivergenceLoss(opt_loss, env)
    else:
        raise NotImplementedError