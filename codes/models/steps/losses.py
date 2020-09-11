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
    elif type == 'interpreted_feature':
        return InterpretedFeatureLoss(opt_loss, env)
    elif type == 'generator_gan':
        return GeneratorGanLoss(opt_loss, env)
    elif type == 'discriminator_gan':
        return DiscriminatorGanLoss(opt_loss, env)
    else:
        raise NotImplementedError


# Converts params to a list of tensors extracted from state. Works with list/tuple params as well as scalars.
def extract_params_from_state(params, state):
    if isinstance(params, list) or isinstance(params, tuple):
        p = [state[r] for r in params]
    else:
        p = [state[params]]
    return p


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
        self.netF = define_F(which_model=opt['which_model_F'],
                             load_path=opt['load_path'] if 'load_path' in opt.keys() else None).to(self.env['device'])
        if not env['opt']['dist']:
            self.netF = torch.nn.parallel.DataParallel(self.netF)

    def forward(self, net, state):
        with torch.no_grad():
            logits_real = self.netF(state[self.opt['real']])
        logits_fake = self.netF(state[self.opt['fake']])
        return self.criterion(logits_fake, logits_real)


# Special form of feature loss which first computes the feature embedding for the truth space, then uses a second
# network which was trained to replicate that embedding on an altered input space (for example, LR or greyscale) to
# compute the embedding in the generated space. Useful for weakening the influence of the feature network in controlled
# ways.
class InterpretedFeatureLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(InterpretedFeatureLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = get_basic_criterion_for_name(opt['criterion'], env['device'])
        self.netF_real = define_F(which_model=opt['which_model_F']).to(self.env['device'])
        self.netF_gen = define_F(which_model=opt['which_model_F'], load_path=opt['load_path']).to(self.env['device'])
        if not env['opt']['dist']:
            self.netF_real = torch.nn.parallel.DataParallel(self.netF_real)
            self.netF_gen = torch.nn.parallel.DataParallel(self.netF_gen)

    def forward(self, net, state):
        logits_real = self.netF_real(state[self.opt['real']])
        logits_fake = self.netF_gen(state[self.opt['fake']])
        return self.criterion(logits_fake, logits_real)


class GeneratorGanLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(GeneratorGanLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = GANLoss(opt['gan_type'], 1.0, 0.0).to(env['device'])

    def forward(self, net, state):
        netD = self.env['discriminators'][self.opt['discriminator']]
        fake = extract_params_from_state(self.opt['fake'], state)
        if self.opt['gan_type'] in ['gan', 'pixgan', 'pixgan_fea']:
            pred_g_fake = netD(*fake)
            return self.criterion(pred_g_fake, True)
        elif self.opt['gan_type'] == 'ragan':
            real = extract_params_from_state(self.opt['real'], state)
            real = [r.detach() for r in real]
            pred_d_real = netD(*real).detach()
            pred_g_fake = netD(*fake)
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
        real = extract_params_from_state(self.opt['real'], state)
        fake = extract_params_from_state(self.opt['fake'], state)
        fake = [f.detach() for f in fake]
        d_real = net(*real)
        d_fake = net(*fake)

        self.metrics.append(("d_fake", torch.mean(d_fake)))
        self.metrics.append(("d_real", torch.mean(d_real)))

        if self.opt['gan_type'] in ['gan', 'pixgan']:
            l_real = self.criterion(d_real, True)
            l_fake = self.criterion(d_fake, False)
            l_total = l_real + l_fake
            return l_total
        elif self.opt['gan_type'] == 'ragan':
            return (self.criterion(d_real - torch.mean(d_fake), True) +
                    self.criterion(d_fake - torch.mean(d_real), False))
        else:
            raise NotImplementedError

