import torch
import torch.nn as nn
from models.networks import define_F
from models.loss import GANLoss


def create_generator_loss(opt_loss, env):
    type = opt_loss['type']
    if type == 'pix':
        return PixLoss(opt_loss, env)
    elif type == 'feature':
        return FeatureLoss(opt_loss, env)
    elif type == 'generator_gan':
        return GeneratorGanLoss(opt_loss, env)
    elif type == 'discriminator_gan':
        return DiscriminatorGanLoss(opt_loss, env)
    else:
        raise NotImplementedError


class ConfigurableLoss(nn.Module):
    def __init__(self, opt, env):
        super(self, ConfigurableLoss).__init__()
        self.opt = opt
        self.env = env

    def forward(self, net, state):
        raise NotImplementedError


def get_basic_criterion_for_name(name, device):
    if name == 'l1':
        return nn.L1Loss(device=device)
    elif name == 'l2':
        return nn.MSELoss(device=device)
    else:
        raise NotImplementedError


class PixLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(self, PixLoss).__init__(opt, env)
        self.opt = opt
        self.criterion = get_basic_criterion_for_name(opt['criterion'], env['device'])

    def forward(self, net, state):
        return self.criterion(state[self.opt['fake']], state[self.opt['real']])


class FeatureLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(self, FeatureLoss).__init__(opt, env)
        self.opt = opt
        self.criterion = get_basic_criterion_for_name(opt['criterion'], env['device'])
        self.netF = define_F(opt).to(self.env['device'])

    def forward(self, net, state):
        with torch.no_grad():
            logits_real = self.netF(state[self.opt['real']])
            logits_fake = self.netF(state[self.opt['fake']])
        return self.criterion(logits_fake, logits_real)


class GeneratorGanLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(self, GeneratorGanLoss).__init__(opt, env)
        self.opt = opt
        self.criterion = GANLoss(opt['gan_type'], 1.0, 0.0).to(env['device'])
        self.netD = env['discriminators'][opt['discriminator']]

    def forward(self, net, state):
        if self.opt['gan_type'] in ['gan', 'pixgan', 'pixgan_fea', 'crossgan']:
            if self.opt['gan_type'] == 'crossgan':
                pred_g_fake = self.netD(state[self.opt['fake']], state['lq'])
            else:
                pred_g_fake = self.netD(state[self.opt['fake']])
            return self.criterion(pred_g_fake, True)
        elif self.opt['gan_type'] == 'ragan':
            pred_d_real = self.netD(state[self.opt['real']]).detach()
            pred_g_fake = self.netD(state[self.opt['fake']])
            return (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                    self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
        else:
            raise NotImplementedError


class DiscriminatorGanLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(self, DiscriminatorGanLoss).__init__(opt, env)
        self.opt = opt
        self.criterion = GANLoss(opt['gan_type'], 1.0, 0.0).to(env['device'])

    def forward(self, net, state):
        if self.opt['gan_type'] in ['gan', 'pixgan', 'pixgan_fea', 'crossgan']:
            if self.opt['gan_type'] == 'crossgan':
                pred_g_fake = net(state[self.opt['fake']].detach(), state['lq'])
            else:
                pred_g_fake = net(state[self.opt['fake']].detach())
            return self.criterion(pred_g_fake, False)
        elif self.opt['gan_type'] == 'ragan':
            pred_d_real = self.netD(state[self.opt['real']])
            pred_g_fake = self.netD(state[self.opt['fake']].detach())
            return (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), True) +
                    self.cri_gan(pred_g_fake - torch.mean(pred_d_real), False)) / 2
        else:
            raise NotImplementedError
