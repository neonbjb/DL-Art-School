import torch
import torch.nn as nn
from models.networks import define_F
from models.loss import GANLoss
from torchvision.utils import save_image


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
        super(ConfigurableLoss, self).__init__()
        self.opt = opt
        self.env = env
        self.metrics = []

    def forward(self, net, state):
        raise NotImplementedError

    def extra_metrics(self):
        return self.metrics


def get_basic_criterion_for_name(name, device):
    if name == 'l1':
        return nn.L1Loss().to(device)
    elif name == 'l2':
        return nn.MSELoss().to(device)
    else:
        raise NotImplementedError


class PixLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(PixLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = get_basic_criterion_for_name(opt['criterion'], env['device'])

    def forward(self, net, state):
        return self.criterion(state[self.opt['fake']], state[self.opt['real']])


class FeatureLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(FeatureLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = get_basic_criterion_for_name(opt['criterion'], env['device'])
        self.netF = define_F(which_model=opt['which_model_F']).to(self.env['device'])
        if not env['opt']['dist']:
            self.netF = torch.nn.parallel.DataParallel(self.netF)

    def forward(self, net, state):
        with torch.no_grad():
            logits_real = self.netF(state[self.opt['real']])
        logits_fake = self.netF(state[self.opt['fake']])
        return self.criterion(logits_fake, logits_real)


class GeneratorGanLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(GeneratorGanLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = GANLoss(opt['gan_type'], 1.0, 0.0).to(env['device'])

    def forward(self, net, state):
        netD = self.env['discriminators'][self.opt['discriminator']]
        if self.opt['gan_type'] in ['gan', 'pixgan', 'pixgan_fea', 'crossgan', 'crossgan_lrref']:
            if self.opt['gan_type'] == 'crossgan':
                pred_g_fake = netD(state[self.opt['fake']], state['lq_fullsize_ref'])
            elif self.opt['gan_type'] == 'crossgan_lrref':
                pred_g_fake = netD(state[self.opt['fake']], state['lq'])
            else:
                pred_g_fake = netD(state[self.opt['fake']])
            return self.criterion(pred_g_fake, True)
        elif self.opt['gan_type'] == 'ragan':
            pred_d_real = netD(state[self.opt['real']]).detach()
            pred_g_fake = netD(state[self.opt['fake']])
            return (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                    self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
        else:
            raise NotImplementedError


class DiscriminatorGanLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(DiscriminatorGanLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = GANLoss(opt['gan_type'], 1.0, 0.0).to(env['device'])

    def forward(self, net, state):
        self.metrics = []

        if self.opt['gan_type'] == 'crossgan':
            d_real = net(state[self.opt['real']], state['lq_fullsize_ref'])
            d_fake = net(state[self.opt['fake']].detach(), state['lq_fullsize_ref'])
            mismatched_lq = torch.roll(state['lq_fullsize_ref'], shifts=1, dims=0)
            d_mismatch_real = net(state[self.opt['real']], mismatched_lq)
            d_mismatch_fake = net(state[self.opt['fake']].detach(), mismatched_lq)
        elif self.opt['gan_type'] == 'crossgan_lrref':
            d_real = net(state[self.opt['real']], state['lq'])
            d_fake = net(state[self.opt['fake']].detach(), state['lq'])
            mismatched_lq = torch.roll(state['lq'], shifts=1, dims=0)
            d_mismatch_real = net(state[self.opt['real']], mismatched_lq)
            d_mismatch_fake = net(state[self.opt['fake']].detach(), mismatched_lq)
        else:
            d_real = net(state[self.opt['real']])
            d_fake = net(state[self.opt['fake']].detach())
        self.metrics.append(("d_fake", torch.mean(d_fake)))
        self.metrics.append(("d_real", torch.mean(d_real)))

        if self.opt['gan_type'] in ['gan', 'pixgan', 'crossgan', 'crossgan_lrref']:
            l_real = self.criterion(d_real, True)
            l_fake = self.criterion(d_fake, False)
            l_total = l_real + l_fake
            if 'crossgan' in self.opt['gan_type']:
                l_mreal = self.criterion(d_mismatch_real, False)
                l_mfake = self.criterion(d_mismatch_fake, False)
                l_total += l_mreal + l_mfake
                self.metrics.append(("l_mismatch", l_mfake + l_mreal))
            self.metrics.append(("l_fake", l_fake))
            self.metrics.append(("l_real", l_real))
            return l_total
        elif self.opt['gan_type'] == 'ragan':
            return (self.criterion(d_real - torch.mean(d_fake), True) +
                    self.criterion(d_fake - torch.mean(d_real), False))
        else:
            raise NotImplementedError

