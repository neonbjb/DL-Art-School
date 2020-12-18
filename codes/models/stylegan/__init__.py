import models.archs.stylegan.stylegan2_lucidrains as stylegan2


def create_stylegan2_loss(opt_loss, env):
    type = opt_loss['type']
    if type == 'stylegan2_divergence':
        return stylegan2.StyleGan2DivergenceLoss(opt_loss, env)
    elif type == 'stylegan2_pathlen':
        return stylegan2.StyleGan2PathLengthLoss(opt_loss, env)
    else:
        raise NotImplementedError