import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from apex import amp

import models.networks as networks
from .base_model import BaseModel
from models.loss import GANLoss
import torchvision.utils as utils
from .archs.SPSR_arch import ImageGradient, ImageGradientNoPadding

logger = logging.getLogger('base')

class SPSRModel(BaseModel):
    def __init__(self, opt):
        super(SPSRModel, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)   # D
            self.netD_grad = networks.define_D(opt).to(self.device)  # D_grad
            self.netG.train()
            self.netD.train()
            self.netD_grad.train()
            self.mega_batch_factor = 1
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            self.mega_batch_factor = train_opt['mega_batch_factor']

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

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters are for WGAN
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
            # Branch_init_iters
            self.branch_pretrain = train_opt['branch_pretrain'] if train_opt['branch_pretrain'] else 0
            self.branch_init_iters = train_opt['branch_init_iters'] if train_opt['branch_init_iters'] else 1

            # gradient_pixel_loss
            if train_opt['gradient_pixel_weight'] > 0:
                self.cri_pix_grad = nn.MSELoss().to(self.device)
                self.l_pix_grad_w = train_opt['gradient_pixel_weight']
            else:
                self.cri_pix_grad = None

            # gradient_gan_loss
            if train_opt['gradient_gan_weight'] > 0:
                self.cri_grad_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
                self.l_gan_grad_w = train_opt['gradient_gan_weight']
            else:
                self.cri_grad_gan = None

            # G_grad pixel loss
            if train_opt['pixel_branch_weight'] > 0:
                l_pix_type = train_opt['pixel_branch_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix_branch = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix_branch = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_branch_w = train_opt['pixel_branch_weight']
            else:
                logger.info('Remove G_grad pixel loss.')
                self.cri_pix_branch = None

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

            optim_params = []
            for k, v in self.netG.named_parameters():  # optimize part of the model

                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)

            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))

            self.optimizers.append(self.optimizer_D)

            # D_grad
            wd_D_grad = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D_grad = torch.optim.Adam(self.netD_grad.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))

            self.optimizers.append(self.optimizer_D_grad)

            # AMP
            [self.netG, self.netD, self.netD_grad], [self.optimizer_G, self.optimizer_D, self.optimizer_D_grad] = \
                amp.initialize([self.netG, self.netD, self.netD_grad],
                               [self.optimizer_G, self.optimizer_D, self.optimizer_D_grad],
                               opt_level=self.amp_level, num_losses=3)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
            self.get_grad = ImageGradient()
            self.get_grad_nopadding = ImageGradientNoPadding()

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = [t.to(self.device) for t in torch.chunk(data['LQ'], chunks=self.mega_batch_factor, dim=0)]

        if need_HR:  # train or val
            self.var_H = [t.to(self.device) for t in torch.chunk(data['GT'], chunks=self.mega_batch_factor, dim=0)]
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = [t.to(self.device) for t in torch.chunk(input_ref.to(self.device), chunks=self.mega_batch_factor, dim=0)]



    def optimize_parameters(self, step):
        # Some generators have variants depending on the current step.
        if hasattr(self.netG.module, "update_for_step"):
            self.netG.module.update_for_step(step, os.path.join(self.opt['path']['models'], ".."))
        if hasattr(self.netD.module, "update_for_step"):
            self.netD.module.update_for_step(step, os.path.join(self.opt['path']['models'], ".."))

        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        for p in self.netD_grad.parameters():
            p.requires_grad = False

        if(self.branch_pretrain):
            if(step < self.branch_init_iters):
                for k,v in self.netG.named_parameters():
                    if 'f_' not in k :
                        v.requires_grad=False
            else:
                for k,v in self.netG.named_parameters():
                    if 'f_' not in k :
                        v.requires_grad=True

        self.optimizer_G.zero_grad()

        self.fake_H_branch = []
        self.fake_H = []
        self.grad_LR = []
        for var_L, var_H, var_ref in zip(self.var_L, self.var_H, self.var_ref):
            fake_H_branch, fake_H, grad_LR = self.netG(var_L)
            self.fake_H_branch.append(fake_H_branch.detach())
            self.fake_H.append(fake_H.detach())
            self.grad_LR.append(grad_LR.detach())

            fake_H_grad = self.get_grad(fake_H)
            var_H_grad = self.get_grad(var_H)
            var_ref_grad = self.get_grad(var_ref)
            var_H_grad_nopadding = self.get_grad_nopadding(var_H)

            l_g_total = 0
            if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                if self.cri_pix:  # pixel loss
                    l_g_pix = self.l_pix_w * self.cri_pix(fake_H, var_H)
                    l_g_total += l_g_pix
                if self.cri_fea:  # feature loss
                    real_fea = self.netF(var_H).detach()
                    fake_fea = self.netF(fake_H)
                    l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                    l_g_total += l_g_fea

                if self.cri_pix_grad: #gradient pixel loss
                    l_g_pix_grad = self.l_pix_grad_w * self.cri_pix_grad(fake_H_grad, var_H_grad)
                    l_g_total += l_g_pix_grad

                if self.cri_pix_branch: #branch pixel loss
                    l_g_pix_grad_branch = self.l_pix_branch_w * self.cri_pix_branch(fake_H_branch, var_H_grad_nopadding)
                    l_g_total += l_g_pix_grad_branch

                if self.l_gan_w > 0:
                    # G gan + cls loss
                    pred_g_fake = self.netD(fake_H)
                    pred_d_real = self.netD(var_ref).detach()

                    l_g_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                            self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                    l_g_total += l_g_gan

                if self.cri_grad_gan:
                    # grad G gan + cls loss
                    pred_g_fake_grad = self.netD_grad(fake_H_grad)
                    pred_d_real_grad = self.netD_grad(var_ref_grad).detach()

                    l_g_gan_grad = self.l_gan_grad_w * (self.cri_grad_gan(pred_d_real_grad - torch.mean(pred_g_fake_grad), False) +
                                                        self.cri_grad_gan(pred_g_fake_grad - torch.mean(pred_d_real_grad), True)) /2
                    l_g_total += l_g_gan_grad

                l_g_total /= self.mega_batch_factor
                with amp.scale_loss(l_g_total, self.optimizer_G, loss_id=0) as l_g_total_scaled:
                    l_g_total_scaled.backward()

        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            self.optimizer_G.step()


        if self.l_gan_w > 0:
            # D
            for p in self.netD.parameters():
                p.requires_grad = True

            self.optimizer_D.zero_grad()
            for var_ref, fake_H in zip(self.var_ref, self.fake_H):
                pred_d_real = self.netD(var_ref)
                pred_d_fake = self.netD(fake_H)  # detach to avoid BP to G
                l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)

                l_d_total = (l_d_real + l_d_fake) / 2

                l_d_total /= self.mega_batch_factor
                with amp.scale_loss(l_d_total, self.optimizer_D, loss_id=1) as l_d_total_scaled:
                    l_d_total_scaled.backward()

            self.optimizer_D.step()

        if self.cri_grad_gan:
            for p in self.netD_grad.parameters():
                p.requires_grad = True

            self.optimizer_D_grad.zero_grad()
            for var_ref, fake_H in zip(self.var_ref, self.fake_H):
                fake_H_grad = self.get_grad(fake_H)
                var_ref_grad = self.get_grad(var_ref)

                pred_d_real_grad = self.netD_grad(var_ref_grad)
                pred_d_fake_grad = self.netD_grad(fake_H_grad.detach())  # detach to avoid BP to G

                l_d_real_grad = self.cri_grad_gan(pred_d_real_grad - torch.mean(pred_d_fake_grad), True)
                l_d_fake_grad = self.cri_grad_gan(pred_d_fake_grad - torch.mean(pred_d_real_grad), False)

                l_d_total_grad = (l_d_real_grad + l_d_fake_grad) / 2
                l_d_total_grad /= self.mega_batch_factor

                with amp.scale_loss(l_d_total_grad, self.optimizer_D_grad, loss_id=2) as l_d_total_grad_scaled:
                    l_d_total_grad_scaled.backward()

            self.optimizer_D_grad.step()

        # Log sample images from first microbatch.
        if step % 50 == 0:
            sample_save_path = os.path.join(self.opt['path']['models'], "..", "temp")
            os.makedirs(os.path.join(sample_save_path, "hr"), exist_ok=True)
            os.makedirs(os.path.join(sample_save_path, "lr"), exist_ok=True)
            os.makedirs(os.path.join(sample_save_path, "gen"), exist_ok=True)
            os.makedirs(os.path.join(sample_save_path, "gen_grad"), exist_ok=True)
            # fed_LQ is not chunked.
            utils.save_image(self.var_H[0].cpu(), os.path.join(sample_save_path, "hr", "%05i.png" % (step,)))
            utils.save_image(self.var_L[0].cpu(), os.path.join(sample_save_path, "lr", "%05i.png" % (step,)))
            utils.save_image(self.fake_H[0].cpu(), os.path.join(sample_save_path, "gen", "%05i.png" % (step,)))
            utils.save_image(self.grad_LR[0].cpu(), os.path.join(sample_save_path, "gen_grad", "%05i.png" % (step,)))


        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # G
            if self.cri_pix:
                self.add_log_entry('l_g_pix', l_g_pix.item())
            if self.cri_fea:
                self.add_log_entry('l_g_fea', l_g_fea.item())
            if self.l_gan_w > 0:
                self.add_log_entry('l_g_gan', l_g_gan.item())

            if self.cri_pix_branch: #branch pixel loss
                self.add_log_entry('l_g_pix_grad_branch', l_g_pix_grad_branch.item())

        if self.l_gan_w > 0:
            self.add_log_entry('l_d_real', l_d_real.item())
            self.add_log_entry('l_d_fake', l_d_fake.item())
            self.add_log_entry('l_d_real_grad', l_d_real_grad.item())
            self.add_log_entry('l_d_fake_grad', l_d_fake_grad.item())
            self.add_log_entry('D_real', torch.mean(pred_d_real.detach()))
            self.add_log_entry('D_fake', torch.mean(pred_d_fake.detach()))
            self.add_log_entry('D_real_grad', torch.mean(pred_d_real_grad.detach()))
            self.add_log_entry('D_fake_grad', torch.mean(pred_d_fake_grad.detach()))

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

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H_branch = []
            self.fake_H = []
            self.grad_LR = []
            for var_L in self.var_L:
                fake_H_branch, fake_H, grad_LR = self.netG(var_L)
                self.fake_H_branch.append(fake_H_branch)
                self.fake_H.append(fake_H)
                self.grad_LR.append(grad_LR)
            
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

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L[0].float().cpu()
        
        out_dict['rlt'] = self.fake_H[0].float().cpu()
        out_dict['SR_branch'] = self.fake_H_branch[0].float().cpu()
        out_dict['LR_grad'] = self.grad_LR[0].float().cpu()
        if need_HR:
            out_dict['GT'] = self.var_H[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:
            # Disriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)

            logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                    self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)

                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD)
        load_path_D_grad = self.opt['path']['pretrain_model_D_grad']
        if self.opt['is_train'] and load_path_D_grad is not None:
            logger.info('Loading pretrained model for D_grad [{:s}] ...'.format(load_path_D_grad))
            self.load_network(load_path_D_grad, self.netD_grad)

    def compute_fea_loss(self, real, fake):
        if self.cri_fea is None:
            return 0
        with torch.no_grad():
            real = real.unsqueeze(dim=0).to(self.device)
            fake = fake.unsqueeze(dim=0).to(self.device)
            real_fea = self.netF(real).detach()
            fake_fea = self.netF(fake)
            return self.cri_fea(fake_fea, real_fea).item()

    def force_restore_swapout(self):
        pass

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
        self.save_network(self.netD_grad, 'D_grad', iter_step)

    # override of load_network that allows loading partial params (like RRDB_PSNR_x4)
    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        pretrained_dict = torch.load(load_path)
        model_dict = network.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        network.load_state_dict(model_dict)