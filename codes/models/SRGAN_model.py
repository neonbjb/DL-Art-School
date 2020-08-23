import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from models.base_model import BaseModel
from models.loss import GANLoss, FDPLLoss
from apex import amp
from data.weight_scheduler import get_scheduler_for_opt
from .archs.SPSR_arch import ImageGradient, ImageGradientNoPadding
import torch.nn.functional as F
import glob
import random

import torchvision.utils as utils
import os

logger = logging.getLogger('base')


class SRGANModel(BaseModel):
    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        self.spsr_enabled = 'spsr' in opt['model']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            if self.spsr_enabled:
                self.netD_grad = networks.define_D(opt).to(self.device)  # D_grad

        if 'network_C' in opt.keys():
            self.netC = networks.define_G(opt, net_key='network_C').to(self.device)
            # The corruptor net is fixed. Lock 'her down.
            self.netC.eval()
            for p in self.netC.parameters():
                p.requires_grad = True
        else:
            self.netC = None
        self.mega_batch_factor = 1
        self.disjoint_data = False

        # define losses, optimizer and scheduler
        if self.is_train:
            self.mega_batch_factor = train_opt['mega_batch_factor']
            if self.mega_batch_factor is None:
                self.mega_batch_factor = 1
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # FDPL loss.
            if 'fdpl_loss' in train_opt.keys():
                fdpl_opt = train_opt['fdpl_loss']
                self.fdpl_weight = fdpl_opt['weight']
                self.fdpl_enabled = self.fdpl_weight > 0
                if self.fdpl_enabled:
                    self.cri_fdpl = FDPLLoss(fdpl_opt['data_mean'], self.device)
            else:
                self.fdpl_enabled = False

            if self.spsr_enabled:
                spsr_opt = train_opt['spsr']
                self.branch_pretrain = spsr_opt['branch_pretrain'] if spsr_opt['branch_pretrain'] else 0
                self.branch_init_iters = spsr_opt['branch_init_iters'] if spsr_opt['branch_init_iters'] else 1
                if spsr_opt['gradient_pixel_weight'] > 0:
                    self.cri_pix_grad = nn.MSELoss().to(self.device)
                    self.l_pix_grad_w = spsr_opt['gradient_pixel_weight']
                else:
                    self.cri_pix_grad = None
                if spsr_opt['gradient_gan_weight'] > 0:
                    self.cri_grad_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
                    self.l_gan_grad_w = spsr_opt['gradient_gan_weight']
                else:
                    self.cri_grad_gan = None
                if spsr_opt['pixel_branch_weight'] > 0:
                    l_pix_type = spsr_opt['pixel_branch_criterion']
                    if l_pix_type == 'l1':
                        self.cri_pix_branch = nn.L1Loss().to(self.device)
                    elif l_pix_type == 'l2':
                        self.cri_pix_branch = nn.MSELoss().to(self.device)
                    else:
                        raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                    self.l_pix_branch_w = spsr_opt['pixel_branch_weight']
                else:
                    logger.info('Remove G_grad pixel loss.')
                    self.cri_pix_branch = None

            # G feature loss
            if train_opt['feature_weight'] and train_opt['feature_weight'] > 0:
                # For backwards compatibility, use a scheduler definition instead. Remove this at some point.
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                sched_params = {
                    'type': 'fixed',
                    'weight': train_opt['feature_weight']
                }
                self.l_fea_sched = get_scheduler_for_opt(sched_params)
            elif train_opt['feature_scheduler']:
                self.l_fea_sched = get_scheduler_for_opt(train_opt['feature_scheduler'])
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(use_bn=False).to(self.device)
                self.lr_netF = None
                if 'lr_fea_path' in train_opt.keys():
                    self.lr_netF = networks.define_F(use_bn=False, load_path=train_opt['lr_fea_path']).to(self.device)
                    self.disjoint_data = True

                if opt['dist']:
                    pass  # do not need to use DistributedDataParallel for netF
                else:
                    self.netF = DataParallel(self.netF)
                    if self.lr_netF:
                        self.lr_netF = DataParallel(self.lr_netF)

            # You can feed in a list of frozen pre-trained discriminators. These are treated the same as feature losses.
            self.fixed_disc_nets = []
            if 'fixed_discriminators' in opt.keys():
                for opt_fdisc in opt['fixed_discriminators'].keys():
                    netFD = networks.define_fixed_D(opt['fixed_discriminators'][opt_fdisc]).to(self.device)
                    if opt['dist']:
                        pass  # do not need to use DistributedDataParallel for netF
                    else:
                        netFD = DataParallel(netFD)
                    self.fixed_disc_nets.append(netFD)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
            self.G_warmup = train_opt['G_warmup'] if train_opt['G_warmup'] else -1
            self.D_noise_theta = train_opt['D_noise_theta_init'] if train_opt['D_noise_theta_init'] else 0
            self.D_noise_final = train_opt['D_noise_final_it'] if train_opt['D_noise_final_it'] else 0
            self.D_noise_theta_floor = train_opt['D_noise_theta_floor'] if train_opt['D_noise_theta_floor'] else 0
            self.corruptor_swapout_steps = train_opt['corruptor_swapout_steps'] if train_opt['corruptor_swapout_steps'] else 500
            self.corruptor_usage_prob = train_opt['corruptor_usage_probability'] if train_opt['corruptor_usage_probability'] else .5

            # optimizers
            # G optimizer
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            if train_opt['lr_scheme'] == 'ProgressiveMultiStepLR':
                optim_params = self.netG.get_param_groups()
            else:
                for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            # D optimizer
            optim_params = []
            for k, v in self.netD.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(optim_params, lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)
            self.disc_optimizers.append(self.optimizer_D)

            if self.spsr_enabled:
                # D_grad optimizer
                optim_params = []
                for k, v in self.netD_grad.named_parameters():  # can optimize for a part of the model
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                # D
                wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
                self.optimizer_D_grad = torch.optim.Adam(optim_params, lr=train_opt['lr_D'],
                                                    weight_decay=wd_D,
                                                    betas=(train_opt['beta1_D'], train_opt['beta2_D']))
                self.optimizers.append(self.optimizer_D_grad)
                self.disc_optimizers.append(self.optimizer_D_grad)

            if self.spsr_enabled:
                self.get_grad = ImageGradient().to(self.device)
                self.get_grad_nopadding = ImageGradientNoPadding().to(self.device)
                [self.netG, self.netD, self.netD_grad, self.get_grad, self.get_grad_nopadding], \
                [self.optimizer_G, self.optimizer_D, self.optimizer_D_grad] = \
                    amp.initialize([self.netG, self.netD, self.netD_grad, self.get_grad, self.get_grad_nopadding],
                                   [self.optimizer_G, self.optimizer_D, self.optimizer_D_grad],
                                   opt_level=self.amp_level, num_losses=3)
            else:
                # AMP
                [self.netG, self.netD], [self.optimizer_G, self.optimizer_D] = \
                    amp.initialize([self.netG, self.netD], [self.optimizer_G, self.optimizer_D], opt_level=self.amp_level, num_losses=3)

            # DataParallel
            if opt['dist']:
                self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()],
                                                    find_unused_parameters=True)
            else:
                self.netG = DataParallel(self.netG)
            if self.is_train:
                if opt['dist']:
                    self.netD = DistributedDataParallel(self.netD,
                                                        device_ids=[torch.cuda.current_device()],
                                                        find_unused_parameters=True)
                else:
                    self.netD = DataParallel(self.netD)
                self.netG.train()
                self.netD.train()
                if self.spsr_enabled:
                    self.netD_grad.train()

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                # This is a recent change. assert to make sure any legacy configs dont find their way here.
                assert 'gen_lr_steps' in train_opt.keys() and 'disc_lr_steps' in train_opt.keys()
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(self.optimizer_G, train_opt['gen_lr_steps'],
                                                     restarts=train_opt['restarts'],
                                                     weights=train_opt['restart_weights'],
                                                     gamma=train_opt['lr_gamma'],
                                                     clear_state=train_opt['clear_state'],
                                                     force_lr=train_opt['force_lr']))
                for o in self.disc_optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(o, train_opt['disc_lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state'],
                                                         force_lr=train_opt['force_lr']))
            elif train_opt['lr_scheme'] == 'ProgressiveMultiStepLR':
                # Only supported when there are two optimizers: G and D.
                assert len(self.optimizers) == 2
                self.schedulers.append(lr_scheduler.ProgressiveMultiStepLR(self.optimizer_G, train_opt['gen_lr_steps'],
                                                                           self.netG.module.get_progressive_starts(),
                                                                           train_opt['lr_gamma']))
                for o in self.disc_optimizers:
                    self.schedulers.append(lr_scheduler.ProgressiveMultiStepLR(o, train_opt['disc_lr_steps'],
                                                                               [0],
                                                                               train_opt['lr_gamma']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

            # Swapout params
            self.swapout_G_freq = train_opt['swapout_G_freq'] if train_opt['swapout_G_freq'] else 0
            self.swapout_G_duration = 0
            self.swapout_D_freq = train_opt['swapout_D_freq'] if train_opt['swapout_D_freq'] else 0
            self.swapout_D_duration = 0
            self.swapout_duration = train_opt['swapout_duration'] if train_opt['swapout_duration'] else 0

            # GAN LQ image params
            self.gan_lq_img_use_prob = train_opt['gan_lowres_use_probability'] if train_opt['gan_lowres_use_probability'] else 0

            self.img_debug_steps = opt['logger']['img_debug_steps'] if 'img_debug_steps' in opt['logger'].keys() else 50

        self.print_network()  # print network
        self.load()  # load G and D if needed
        self.load_random_corruptor()

        # Setting this to false triggers SRGAN to call the models update_model() function on the first iteration.
        self.updated = True

    def feed_data(self, data, need_GT=True):
        _profile = True
        if _profile:
            from time import time
            _t = time()

        # Corrupt the data with the given corruptor, if specified.
        self.fed_LQ = data['LQ'].to(self.device)
        if self.netC and random.random() < self.corruptor_usage_prob:
            with torch.no_grad():
                corrupted_L = self.netC(self.fed_LQ)[0].detach()
        else:
            corrupted_L = self.fed_LQ

        self.var_L = torch.chunk(corrupted_L, chunks=self.mega_batch_factor, dim=0)
        if need_GT:
            self.var_H = [t.to(self.device) for t in torch.chunk(data['GT'], chunks=self.mega_batch_factor, dim=0)]
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = [t.to(self.device) for t in torch.chunk(input_ref, chunks=self.mega_batch_factor, dim=0)]
            self.pix = [t.to(self.device) for t in torch.chunk(data['PIX'], chunks=self.mega_batch_factor, dim=0)]

        if 'GAN' in data.keys():
            self.gan_img = [t.to(self.device) for t in torch.chunk(data['GAN'], chunks=self.mega_batch_factor, dim=0)]
        else:
            # If not provided, use provided LQ for anyplace where the GAN would have been used.
            self.gan_img = self.var_L

        if not self.updated:
            self.netG.module.update_model(self.optimizer_G, self.schedulers[0])
            self.updated = True

    def optimize_parameters(self, step):
        _profile = False
        if _profile:
            from time import time
            _t = time()

        # Some generators have variants depending on the current step.
        if hasattr(self.netG.module, "update_for_step"):
            self.netG.module.update_for_step(step, os.path.join(self.opt['path']['models'], ".."))
        if hasattr(self.netD.module, "update_for_step"):
            self.netD.module.update_for_step(step, os.path.join(self.opt['path']['models'], ".."))

        # G
        for p in self.netD.parameters():
            p.requires_grad = False
        if self.spsr_enabled:
            for p in self.netD_grad.parameters():
                p.requires_grad = False

        self.swapout_D(step)
        self.swapout_G(step)

        # Turning off G-grad is required to enable mega-batching and D_update_ratio to work together for some reason.
        if step % self.D_update_ratio == 0 and step >= self.D_init_iters:
            if self.spsr_enabled and self.branch_pretrain and step < self.branch_init_iters:
                for k, v in self.netG.named_parameters():
                    if v.dtype != torch.int64 and v.dtype != torch.bool:
                        v.requires_grad = '_branch_pretrain' in k
            else:
                for p in self.netG.parameters():
                    if p.dtype != torch.int64 and p.dtype != torch.bool:
                        p.requires_grad = True
        else:
            for p in self.netG.parameters():
                p.requires_grad = False

        # Calculate a standard deviation for the gaussian noise to be applied to the discriminator, termed noise-theta.
        if self.D_noise_final == 0:
            noise_theta = 0
        else:
            noise_theta = (self.D_noise_theta - self.D_noise_theta_floor) * (self.D_noise_final - min(step, self.D_noise_final)) / self.D_noise_final + self.D_noise_theta_floor

        if _profile:
            print("Misc setup %f" % (time() - _t,))
            _t = time()

        if step >= self.init_iters:
            self.optimizer_G.zero_grad()
        self.fake_GenOut = []
        self.fea_GenOut = []
        self.fake_H = []
        self.spsr_grad_GenOut = []
        var_ref_skips = []
        for var_L, var_LGAN, var_H, var_ref, pix in zip(self.var_L, self.gan_img, self.var_H, self.var_ref, self.pix):
            if self.spsr_enabled:
                using_gan_img = False
                # SPSR models have outputs from three different branches.
                fake_H_branch, fake_GenOut, grad_LR = self.netG(var_L)
                fea_GenOut = fake_GenOut
                self.spsr_grad_GenOut.append(fake_H_branch)
                # Get image gradients for later use.
                fake_H_grad = self.get_grad_nopadding(fake_GenOut)
            else:
                if random.random() > self.gan_lq_img_use_prob:
                    fea_GenOut, fake_GenOut = self.netG(var_L)
                    using_gan_img = False
                else:
                    fea_GenOut, fake_GenOut = self.netG(var_LGAN)
                    using_gan_img = True

            if _profile:
                print("Gen forward %f" % (time() - _t,))
                _t = time()

            self.fake_GenOut.append(fake_GenOut.detach())
            self.fea_GenOut.append(fea_GenOut.detach())

            l_g_total = 0
            if step % self.D_update_ratio == 0 and step >= self.D_init_iters:
                fea_w = self.l_fea_sched.get_weight_for_step(step)
                l_g_pix_log = None
                l_g_fea_log = None
                l_g_fdpl = None
                l_g_fea_log = None
                if self.cri_pix and not using_gan_img:  # pixel loss
                    l_g_pix = self.l_pix_w * self.cri_pix(fea_GenOut, pix)
                    l_g_pix_log = l_g_pix / self.l_pix_w
                    l_g_total += l_g_pix
                if self.spsr_enabled and self.cri_pix_grad:  # gradient pixel loss
                    if self.disjoint_data:
                        grad_truth = self.get_grad_nopadding(var_L)
                        grad_pred = F.interpolate(fake_H_grad, size=grad_truth.shape[2:], mode="nearest")
                    else:
                        grad_truth = self.get_grad_nopadding(var_H)
                        grad_pred = fake_H_grad
                    l_g_pix_grad = self.l_pix_grad_w * self.cri_pix_grad(grad_pred, grad_truth)
                    l_g_total += l_g_pix_grad
                if self.spsr_enabled and self.cri_pix_branch:  # branch pixel loss
                    if self.disjoint_data:
                        grad_truth = self.get_grad_nopadding(var_L)
                        grad_pred = F.interpolate(fake_H_branch, size=grad_truth.shape[2:], mode="nearest")
                    else:
                        grad_truth = self.get_grad_nopadding(var_H)
                        grad_pred = fake_H_branch
                    l_g_pix_grad_branch = self.l_pix_branch_w * self.cri_pix_branch(grad_pred, grad_truth)
                    l_g_total += l_g_pix_grad_branch
                if self.fdpl_enabled and not using_gan_img:
                    l_g_fdpl = self.cri_fdpl(fea_GenOut, pix)
                    l_g_total += l_g_fdpl * self.fdpl_weight
                if self.cri_fea and not using_gan_img and fea_w > 0:  # feature loss
                    if self.lr_netF is not None:
                        real_fea = self.lr_netF(var_L, interpolate_factor=self.opt['scale'])
                    else:
                        real_fea = self.netF(pix).detach()
                    fake_fea = self.netF(fea_GenOut)
                    l_g_fea = fea_w * self.cri_fea(fake_fea, real_fea)
                    l_g_fea_log = l_g_fea / fea_w
                    l_g_total += l_g_fea

                    if _profile:
                        print("Fea forward %f" % (time() - _t,))
                        _t = time()

                    # Note to future self: The BCELoss(0, 1) and BCELoss(0, 0) = .6931
                    # Effectively this means that the generator has only completely "won" when l_d_real and l_d_fake is
                    # equal to this value. If I ever come up with an algorithm that tunes fea/gan weights automatically,
                    # it should target this

                l_g_fix_disc = torch.zeros(1, requires_grad=False, device=self.device).squeeze()
                for fixed_disc in self.fixed_disc_nets:
                    weight = fixed_disc.module.fdisc_weight
                    real_fea = fixed_disc(pix).detach()
                    fake_fea = fixed_disc(fea_GenOut)
                    l_g_fix_disc = l_g_fix_disc + weight * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fix_disc


                if self.l_gan_w > 0:
                    if self.opt['train']['gan_type'] in ['gan', 'pixgan', 'pixgan_fea', 'crossgan']:
                        if self.opt['train']['gan_type'] == 'crossgan':
                            pred_g_fake = self.netD(fake_GenOut, var_L)
                        else:
                            pred_g_fake = self.netD(fake_GenOut)
                        l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
                    elif self.opt['train']['gan_type'] == 'ragan':
                        pred_d_real = self.netD(var_ref).detach()
                        pred_g_fake = self.netD(fake_GenOut)
                        l_g_gan = self.l_gan_w * (
                            self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                            self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                    l_g_gan_log = l_g_gan / self.l_gan_w
                    l_g_total += l_g_gan

                if self.spsr_enabled and self.cri_grad_gan:
                    if self.opt['train']['gan_type'] == 'crossgan':
                        pred_g_fake_grad = self.netD_grad(fake_H_grad, var_L)
                        pred_g_fake_grad_branch = self.netD_grad(fake_H_branch, var_L)
                    else:
                        pred_g_fake_grad = self.netD_grad(fake_H_grad)
                        pred_g_fake_grad_branch = self.netD_grad(fake_H_branch)
                    if self.opt['train']['gan_type'] in ['gan', 'pixgan', 'pixgan_fea', 'crossgan']:
                        l_g_gan_grad = self.l_gan_grad_w * self.cri_grad_gan(pred_g_fake_grad, True)
                        l_g_gan_grad_branch = self.l_gan_grad_w * self.cri_grad_gan(pred_g_fake_grad_branch, True)
                    elif self.opt['train']['gan_type'] == 'ragan':
                        pred_g_real_grad = self.netD_grad(self.get_grad_nopadding(var_ref)).detach()
                        l_g_gan_grad = self.l_gan_grad_w * (
                            self.cri_gan(pred_g_real_grad - torch.mean(pred_g_fake_grad), False) +
                            self.cri_gan(pred_g_fake_grad - torch.mean(pred_g_real_grad), True)) / 2
                        l_g_gan_grad_branch = self.l_gan_grad_w * (
                            self.cri_gan(pred_g_real_grad - torch.mean(pred_g_fake_grad_branch), False) +
                            self.cri_gan(pred_g_fake_grad_branch - torch.mean(pred_g_real_grad), True)) / 2
                    l_g_total += l_g_gan_grad + l_g_gan_grad_branch

                # Scale the loss down by the batch factor.
                l_g_total_log = l_g_total
                l_g_total = l_g_total / self.mega_batch_factor

                with amp.scale_loss(l_g_total, self.optimizer_G, loss_id=0) as l_g_total_scaled:
                    l_g_total_scaled.backward()

                if _profile:
                    print("Gen backward %f" % (time() - _t,))
                    _t = time()

        self.optimizer_G.step()

        if _profile:
            print("Gen step %f" % (time() - _t,))
            _t = time()

        # D
        if self.l_gan_w > 0 and step >= self.G_warmup:
            for p in self.netD.parameters():
                if p.dtype != torch.int64 and p.dtype != torch.bool:
                    p.requires_grad = True

            noise = torch.randn_like(var_ref) * noise_theta
            noise.to(self.device)
            real_disc_images = []
            fake_disc_images = []
            for fake_GenOut, var_LGAN, var_L, var_H, var_ref, pix in zip(self.fake_GenOut, self.gan_img, self.var_L, self.var_H, self.var_ref, self.pix):
                if random.random() > self.gan_lq_img_use_prob:
                    fake_H = fake_GenOut.clone().detach().requires_grad_(False)
                else:
                    # Re-compute generator outputs with the GAN inputs.
                    with torch.no_grad():
                        if self.spsr_enabled:
                            _, fake_H, _ = self.netG(var_LGAN)
                        else:
                            _, fake_H = self.netG(var_LGAN)
                        fake_H = fake_H.detach()

                        if _profile:
                            print("Gen forward for disc %f" % (time() - _t,))
                            _t = time()

                # Apply noise to the inputs to slow discriminator convergence.
                var_ref = var_ref + noise
                fake_H = fake_H + noise
                l_d_fea_real = 0
                l_d_fea_fake = 0
                self.optimizer_D.zero_grad()
                if self.opt['train']['gan_type'] == 'pixgan_fea':
                    # Compute a feature loss which is added to the GAN loss computed later to guide the discriminator better.
                    disc_fea_scale = .1
                    _, fea_real = self.netD(var_ref, output_feature_vector=True)
                    actual_fea = self.netF(var_ref)
                    l_d_fea_real = self.cri_fea(fea_real, actual_fea) * disc_fea_scale / self.mega_batch_factor
                    _, fea_fake = self.netD(fake_H, output_feature_vector=True)
                    actual_fea = self.netF(fake_H)
                    l_d_fea_fake = self.cri_fea(fea_fake, actual_fea) * disc_fea_scale / self.mega_batch_factor
                if self.opt['train']['gan_type'] == 'crossgan':
                    # need to forward and backward separately, since batch norm statistics differ
                    # real
                    pred_d_real = self.netD(var_ref, var_L)
                    l_d_real = self.cri_gan(pred_d_real, True)
                    l_d_real_log = l_d_real
                    # fake
                    pred_d_fake = self.netD(fake_H, var_L)
                    l_d_fake = self.cri_gan(pred_d_fake, False)
                    l_d_fake_log = l_d_fake
                    # mismatched
                    mismatched_L = torch.roll(var_L, shifts=1, dims=0)
                    pred_d_real_mismatched = self.netD(var_ref, mismatched_L)
                    pred_d_fake_mismatched = self.netD(fake_H, mismatched_L)
                    l_d_mismatched = (self.cri_gan(pred_d_real_mismatched, False) + self.cri_gan(pred_d_fake_mismatched, False)) / 2

                    l_d_total = (l_d_real + l_d_fake + l_d_mismatched) / 3
                    l_d_total = l_d_total / self.mega_batch_factor
                    with amp.scale_loss(l_d_total, self.optimizer_D, loss_id=1) as l_d_total_scaled:
                        l_d_total_scaled.backward()
                elif self.opt['train']['gan_type'] == 'gan':
                    # real
                    pred_d_real = self.netD(var_ref)
                    l_d_real = self.cri_gan(pred_d_real, True) / self.mega_batch_factor
                    l_d_real_log = l_d_real * self.mega_batch_factor
                    # fake
                    pred_d_fake = self.netD(fake_H)
                    l_d_fake = self.cri_gan(pred_d_fake, False) / self.mega_batch_factor
                    l_d_fake_log = l_d_fake * self.mega_batch_factor

                    l_d_total = (l_d_real + l_d_fake) / 2
                    with amp.scale_loss(l_d_total, self.optimizer_D, loss_id=1) as l_d_total_scaled:
                        l_d_total_scaled.backward()
                elif 'pixgan' in self.opt['train']['gan_type']:
                    pixdisc_channels, pixdisc_output_reduction = self.netD.module.pixgan_parameters()
                    disc_output_shape = (var_ref.shape[0], pixdisc_channels, var_ref.shape[2] // pixdisc_output_reduction, var_ref.shape[3] // pixdisc_output_reduction)
                    b, _, w, h = var_ref.shape
                    real = torch.ones((b, pixdisc_channels, w, h), device=var_ref.device)
                    fake = torch.zeros((b, pixdisc_channels, w, h), device=var_ref.device)
                    if not self.disjoint_data:
                        # randomly determine portions of the image to swap to keep the discriminator honest.
                        SWAP_MAX_DIM = w // 4
                        SWAP_MIN_DIM = 16
                        assert SWAP_MAX_DIM > 0
                        if random.random() > .5:   # Make this only happen half the time. Earlier experiments had it happen
                                                   # more often and the model was "cheating" by using the presence of
                                                   # easily discriminated fake swaps to count the entire generated image
                                                   # as fake.
                            random_swap_count = random.randint(0, 4)
                            for i in range(random_swap_count):
                                # Make the swap across fake_H and var_ref
                                swap_x, swap_y = random.randint(0, w - SWAP_MIN_DIM), random.randint(0, h - SWAP_MIN_DIM)
                                swap_w, swap_h = random.randint(SWAP_MIN_DIM, SWAP_MAX_DIM), random.randint(SWAP_MIN_DIM, SWAP_MAX_DIM)
                                if swap_x + swap_w > w:
                                    swap_w = w - swap_x
                                if swap_y + swap_h > h:
                                    swap_h = h - swap_y
                                t = fake_H[:, :, swap_x:(swap_x+swap_w), swap_y:(swap_y+swap_h)].clone()
                                fake_H[:, :, swap_x:(swap_x+swap_w), swap_y:(swap_y+swap_h)] = var_ref[:, :, swap_x:(swap_x+swap_w), swap_y:(swap_y+swap_h)]
                                var_ref[:, :, swap_x:(swap_x+swap_w), swap_y:(swap_y+swap_h)] = t
                                real[:, :, swap_x:(swap_x+swap_w), swap_y:(swap_y+swap_h)] = 0.0
                                fake[:, :, swap_x:(swap_x+swap_w), swap_y:(swap_y+swap_h)] = 1.0

                    # Interpolate down to the dimensionality that the discriminator uses.
                    real = F.interpolate(real, size=disc_output_shape[2:], mode="bilinear", align_corners=False)
                    fake = F.interpolate(fake, size=disc_output_shape[2:], mode="bilinear", align_corners=False)

                    # We're also assuming that this is exactly how the flattened discriminator output is generated.
                    real = real.view(-1, 1)
                    fake = fake.view(-1, 1)

                    # real
                    pred_d_real = self.netD(var_ref)
                    l_d_real = self.cri_gan(pred_d_real, real) / self.mega_batch_factor
                    l_d_real_log = l_d_real * self.mega_batch_factor
                    l_d_real += l_d_fea_real
                    # fake
                    pred_d_fake = self.netD(fake_H)
                    l_d_fake = self.cri_gan(pred_d_fake, fake) / self.mega_batch_factor
                    l_d_fake_log = l_d_fake * self.mega_batch_factor
                    l_d_fake += l_d_fea_fake

                    l_d_total = (l_d_real + l_d_fake) / 2
                    with amp.scale_loss(l_d_total, self.optimizer_D, loss_id=1) as l_d_total_scaled:
                        l_d_total_scaled.backward()

                    pdr = pred_d_real.detach() + torch.abs(torch.min(pred_d_real))
                    pdr = pdr / torch.max(pdr)
                    real_disc_images.append(pdr.view(disc_output_shape))
                    pdf = pred_d_fake.detach() + torch.abs(torch.min(pred_d_fake))
                    pdf = pdf / torch.max(pdf)
                    fake_disc_images.append(pdf.view(disc_output_shape))
                elif self.opt['train']['gan_type'] == 'ragan':
                    pred_d_fake = self.netD(fake_H)
                    pred_d_real = self.netD(var_ref)
                    l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                    l_d_real_log = l_d_real
                    l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
                    l_d_fake_log = l_d_fake
                    l_d_total = (l_d_real + l_d_fake) / 2
                    l_d_total /= self.mega_batch_factor
                    with amp.scale_loss(l_d_total, self.optimizer_D, loss_id=1) as l_d_total_scaled:
                        l_d_total_scaled.backward()
                var_ref_skips.append(var_ref.detach())
                self.fake_H.append(fake_H.detach())
            self.optimizer_D.step()

            if _profile:
                print("Disc step %f" % (time() - _t,))
                _t = time()

            # D_grad.
            if self.spsr_enabled and self.cri_grad_gan and step >= self.G_warmup:
                for p in self.netD_grad.parameters():
                    p.requires_grad = True
                self.optimizer_D_grad.zero_grad()
                for var_L, var_ref, fake_H, fake_H_grad_branch in zip(self.var_L, var_ref_skips, self.fake_H, self.spsr_grad_GenOut):
                    fake_H_grad = self.get_grad_nopadding(fake_H).detach()
                    var_ref_grad = self.get_grad_nopadding(var_ref)
                    fake_H_grad_branch = fake_H_grad_branch.detach() + noise
                    if self.opt['train']['gan_type'] == 'crossgan':
                        pred_d_real_grad = self.netD_grad(var_ref_grad, var_L)
                        pred_d_fake_grad = self.netD_grad(fake_H_grad, var_L)   # Tensor already detached above.
                        # var_ref and fake_H already has noise added to it. We **must** add noise to fake_H_grad_branch too.
                        pred_d_fake_grad_branch = self.netD_grad(fake_H_grad_branch, var_L)
                    else:
                        pred_d_real_grad = self.netD_grad(var_ref_grad)
                        pred_d_fake_grad = self.netD_grad(fake_H_grad)   # Tensor already detached above.
                        # var_ref and fake_H already has noise added to it. We **must** add noise to fake_H_grad_branch too.
                        pred_d_fake_grad_branch = self.netD_grad(fake_H_grad_branch)
                    if self.opt['train']['gan_type'] == 'gan' or self.opt['train']['gan_type'] == 'crossgan':
                        l_d_real_grad = self.cri_gan(pred_d_real_grad, True)
                        l_d_fake_grad = (self.cri_gan(pred_d_fake_grad, False) + self.cri_gan(pred_d_fake_grad_branch, False)) / 2
                    elif self.opt['train']['gan_type'] == 'pixgan':
                        real = torch.ones_like(pred_d_real_grad)
                        fake = torch.zeros_like(pred_d_fake_grad)
                        l_d_real_grad = self.cri_grad_gan(pred_d_real_grad, real)
                        l_d_fake_grad = (self.cri_grad_gan(pred_d_fake_grad, fake) + \
                                        self.cri_grad_gan(pred_d_fake_grad_branch, fake)) / 2
                    elif self.opt['train']['gan_type'] == 'ragan':
                        l_d_real_grad = self.cri_grad_gan(pred_d_real_grad - torch.mean(pred_d_fake_grad), True)
                        l_d_fake_grad = (self.cri_grad_gan(pred_d_fake_grad - torch.mean(pred_d_real_grad), False) + \
                                        self.cri_grad_gan(pred_d_fake_grad_branch - torch.mean(pred_d_real_grad), False)) / 2

                    l_d_total_grad = (l_d_real_grad + l_d_fake_grad) / 2
                    l_d_total_grad /= self.mega_batch_factor
                    with amp.scale_loss(l_d_total_grad, self.optimizer_D_grad, loss_id=2) as l_d_total_grad_scaled:
                        l_d_total_grad_scaled.backward()
                self.optimizer_D_grad.step()


        # Log sample images from first microbatch.
        if step % self.img_debug_steps == 0:
            sample_save_path = os.path.join(self.opt['path']['models'], "..", "temp")
            os.makedirs(os.path.join(sample_save_path, "hr"), exist_ok=True)
            os.makedirs(os.path.join(sample_save_path, "lr"), exist_ok=True)
            os.makedirs(os.path.join(sample_save_path, "gen_fea"), exist_ok=True)
            os.makedirs(os.path.join(sample_save_path, "gen"), exist_ok=True)
            os.makedirs(os.path.join(sample_save_path, "disc_fake"), exist_ok=True)
            os.makedirs(os.path.join(sample_save_path, "pix"), exist_ok=True)
            os.makedirs(os.path.join(sample_save_path, "disc"), exist_ok=True)
            os.makedirs(os.path.join(sample_save_path, "ref"), exist_ok=True)
            if self.spsr_enabled:
                os.makedirs(os.path.join(sample_save_path, "gen_grad"), exist_ok=True)

            # fed_LQ is not chunked.
            for i in range(self.mega_batch_factor):
                utils.save_image(self.var_H[i].cpu(), os.path.join(sample_save_path, "hr", "%05i_%02i.png" % (step, i)))
                utils.save_image(self.var_L[i].cpu(), os.path.join(sample_save_path, "lr", "%05i_%02i.png" % (step, i)))
                utils.save_image(self.pix[i].cpu(), os.path.join(sample_save_path, "pix", "%05i_%02i.png" % (step, i)))
                utils.save_image(self.fake_GenOut[i].cpu(), os.path.join(sample_save_path, "gen", "%05i_%02i.png" % (step, i)))
                utils.save_image(self.fea_GenOut[i].cpu(), os.path.join(sample_save_path, "gen_fea", "%05i_%02i.png" % (step, i)))
                if self.spsr_enabled:
                    utils.save_image(self.spsr_grad_GenOut[i].cpu(), os.path.join(sample_save_path, "gen_grad", "%05i_%02i.png" % (step, i)))
                if self.l_gan_w > 0 and step >= self.G_warmup and 'pixgan' in self.opt['train']['gan_type']:
                    utils.save_image(var_ref_skips[i].cpu(), os.path.join(sample_save_path, "ref", "%05i_%02i.png" % (step, i)))
                    utils.save_image(self.fake_H[i], os.path.join(sample_save_path, "disc_fake", "fake%05i_%02i.png" % (step, i)))
                    utils.save_image(F.interpolate(fake_disc_images[i], scale_factor=4), os.path.join(sample_save_path, "disc", "fake%05i_%02i.png" % (step, i)))
                    utils.save_image(F.interpolate(real_disc_images[i], scale_factor=4), os.path.join(sample_save_path, "disc", "real%05i_%02i.png" % (step, i)))

        # Log metrics
        if step % self.D_update_ratio == 0 and step >= self.D_init_iters:
            if self.cri_pix and l_g_pix_log is not None:
                self.add_log_entry('l_g_pix', l_g_pix_log.detach().item())
            if self.fdpl_enabled and l_g_fdpl is not None:
                self.add_log_entry('l_g_fdpl', l_g_fdpl.detach().item())
            if self.cri_fea and l_g_fea_log is not None:
                self.add_log_entry('feature_weight', fea_w)
                self.add_log_entry('l_g_fea', l_g_fea_log.detach().item())
                self.add_log_entry('l_g_fix_disc', l_g_fix_disc.detach().item())
            if self.l_gan_w > 0:
                self.add_log_entry('l_g_gan', l_g_gan_log.detach().item())
            self.add_log_entry('l_g_total', l_g_total_log.detach().item())
            if self.opt['train']['gan_type'] == 'pixgan_fea':
                self.add_log_entry('l_d_fea_fake', l_d_fea_fake.detach().item() * self.mega_batch_factor)
                self.add_log_entry('l_d_fea_real', l_d_fea_real.detach().item() * self.mega_batch_factor)
                self.add_log_entry('l_d_fake_total', l_d_fake.detach().item() * self.mega_batch_factor)
                self.add_log_entry('l_d_real_total', l_d_real.detach().item() * self.mega_batch_factor)
            if self.opt['train']['gan_type'] == 'crossgan':
                self.add_log_entry('l_d_mismatched', l_d_mismatched.detach().item())
            if self.spsr_enabled:
                if self.cri_pix_grad:
                    self.add_log_entry('l_g_pix_grad_branch', l_g_pix_grad.detach().item())
                if self.cri_pix_branch:
                    self.add_log_entry('l_g_pix_grad_branch', l_g_pix_grad_branch.detach().item())
                if self.cri_grad_gan:
                    self.add_log_entry('l_g_gan_grad', l_g_gan_grad.detach().item() / self.l_gan_grad_w)
                    self.add_log_entry('l_g_gan_grad_branch', l_g_gan_grad_branch.detach().item() / self.l_gan_grad_w)
        if self.l_gan_w > 0 and step >= self.G_warmup:
            self.add_log_entry('l_d_real', l_d_real_log.detach().item())
            self.add_log_entry('l_d_fake', l_d_fake_log.detach().item())
            self.add_log_entry('D_fake', torch.mean(pred_d_fake.detach()))
            self.add_log_entry('D_diff', torch.mean(pred_d_fake.detach()) - torch.mean(pred_d_real.detach()))
            if self.spsr_enabled:
                self.add_log_entry('l_d_real_grad', l_d_real_grad.detach().item())
                self.add_log_entry('l_d_fake_grad', l_d_fake_grad.detach().item())
                self.add_log_entry('D_fake_grad', torch.mean(pred_d_fake_grad.detach()))
                self.add_log_entry('D_diff_grad', torch.mean(pred_d_fake_grad.detach()) - torch.mean(pred_d_real_grad.detach()))

        # Log learning rates.
        for i, pg in enumerate(self.optimizer_G.param_groups):
            self.add_log_entry('gen_lr_%i' % (i,), pg['lr'])
        for i, pg in enumerate(self.optimizer_D.param_groups):
            self.add_log_entry('disc_lr_%i' % (i,), pg['lr'])

        if step % self.corruptor_swapout_steps == 0 and step > 0:
            self.load_random_corruptor()

    # Allows the log to serve as an easy-to-use rotating buffer.
    def add_log_entry(self, key, value):
        key_it = "%s_it" % (key,)
        log_rotating_buffer_size = 50
        if key not in self.log_dict.keys():
            self.log_dict[key] = []
            self.log_dict[key_it] = 0
        if len(self.log_dict[key]) < log_rotating_buffer_size:
            self.log_dict[key].append(value)
        else:
            self.log_dict[key][self.log_dict[key_it] % log_rotating_buffer_size] = value
        self.log_dict[key_it] += 1

    def pick_rand_prev_model(self, model_suffix):
        previous_models = glob.glob(os.path.join(self.opt['path']['models'], "*_%s.pth" % (model_suffix,)))
        if len(previous_models) <= 1:
            return None
        # Just a note: this intentionally includes the swap model in the list of possibilities.
        return previous_models[random.randint(0, len(previous_models)-1)]

    def compute_fea_loss(self, real, fake):
        with torch.no_grad():
            real = real.unsqueeze(dim=0).to(self.device)
            fake = fake.unsqueeze(dim=0).to(self.device)
            real_fea = self.netF(real).detach()
            fake_fea = self.netF(fake)
            return self.cri_fea(fake_fea, real_fea).item()

    # Called before verification/checkpoint to ensure we're using the real models and not a swapout variant.
    def force_restore_swapout(self):
        if self.swapout_D_duration > 0:
            logger.info("Swapping back to current D model: %s" % (self.stashed_D,))
            self.load_network(self.stashed_D, self.netD, self.opt['path']['strict_load'])
            self.stashed_D = None
            self.swapout_D_duration = 0
        if self.swapout_G_duration > 0:
            logger.info("Swapping back to current G model: %s" % (self.stashed_G,))
            self.load_network(self.stashed_G, self.netG, self.opt['path']['strict_load'])
            self.stashed_G = None
            self.swapout_G_duration = 0

    def swapout_D(self, step):
        if self.swapout_D_duration > 0:
            self.swapout_D_duration -= 1
            if self.swapout_D_duration == 0:
                # Swap back.
                logger.info("Swapping back to current D model: %s" % (self.stashed_D,))
                self.load_network(self.stashed_D, self.netD, self.opt['path']['strict_load'])
                self.stashed_D = None
        elif self.swapout_D_freq != 0 and step % self.swapout_D_freq == 0:
            swapped_model = self.pick_rand_prev_model('D')
            if swapped_model is not None:
                logger.info("Swapping to previous D model: %s" % (swapped_model,))
                self.stashed_D = self.save_network(self.netD, 'D', 'swap_model')
                self.load_network(swapped_model, self.netD, self.opt['path']['strict_load'])
                self.swapout_D_duration = self.swapout_duration

    def swapout_G(self, step):
        if self.swapout_G_duration > 0:
            self.swapout_G_duration -= 1
            if self.swapout_G_duration == 0:
                # Swap back.
                logger.info("Swapping back to current G model: %s" % (self.stashed_G,))
                self.load_network(self.stashed_G, self.netG, self.opt['path']['strict_load'])
                self.stashed_G = None
        elif self.swapout_G_freq != 0 and step % self.swapout_G_freq == 0:
            swapped_model = self.pick_rand_prev_model('G')
            if swapped_model is not None:
                logger.info("Swapping to previous G model: %s" % (swapped_model,))
                self.stashed_G = self.save_network(self.netG, 'G', 'swap_model')
                self.load_network(swapped_model, self.netG, self.opt['path']['strict_load'])
                self.swapout_G_duration = self.swapout_duration

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.spsr_enabled:
                self.fake_H_branch = []
                self.fake_GenOut = []
                self.grad_LR = []
                fake_H_branch, fake_GenOut, grad_LR = self.netG(self.var_L[0])
                self.fake_H_branch.append(fake_H_branch)
                self.fake_GenOut.append(fake_GenOut)
                self.grad_LR.append(grad_LR)
            else:
                self.fake_GenOut = [self.netG(self.var_L[0])]
        self.netG.train()

    # Fetches a summary of the log.
    def get_current_log(self, step):
        return_log = {}
        for k in self.log_dict.keys():
            if not isinstance(self.log_dict[k], list):
                continue
            return_log[k] = sum(self.log_dict[k]) / len(self.log_dict[k])

        # Some generators can do their own metric logging.
        if hasattr(self.netG.module, "get_debug_values"):
            return_log.update(self.netG.module.get_debug_values(step))
        if hasattr(self.netD.module, "get_debug_values"):
            return_log.update(self.netD.module.get_debug_values(step))

        return return_log

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L[0].detach().float().cpu()
        gen_batch = self.fake_GenOut[0]
        if isinstance(gen_batch, tuple):
            gen_batch = gen_batch[0]
        out_dict['rlt'] = gen_batch.detach().float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H[0].detach().float().cpu()
        if self.spsr_enabled:
            out_dict['SR_branch'] = self.fake_H_branch[0].float().cpu()
            out_dict['LR_grad'] = self.grad_LR[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])
        if self.spsr_enabled:
            load_path_D_grad = self.opt['path']['pretrain_model_D_grad']
            if self.opt['is_train'] and load_path_D_grad is not None:
                logger.info('Loading pretrained model for D_grad [{:s}] ...'.format(load_path_D_grad))
                self.load_network(load_path_D_grad, self.netD_grad)

    def load_random_corruptor(self):
        if self.netC is None:
            return
        corruptor_files = glob.glob(os.path.join(self.opt['path']['pretrained_corruptors_dir'], "*.pth"))
        corruptor_to_load = corruptor_files[random.randint(0, len(corruptor_files)-1)]
        logger.info('Swapping corruptor to: %s' % (corruptor_to_load,))
        self.load_network(corruptor_to_load, self.netC, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
        if self.spsr_enabled:
            self.save_network(self.netD_grad, 'D_grad', iter_step)
