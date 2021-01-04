from models.styled_sr.discriminator import StyleSrGanDivergenceLoss


def create_stylesr_loss(opt_loss, env):
    type = opt_loss['type']
    if type == 'style_sr_gan_divergence_loss':
        return StyleSrGanDivergenceLoss(opt_loss, env)
    else:
        raise NotImplementedError
