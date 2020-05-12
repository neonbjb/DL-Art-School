import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from models.base_model import BaseModel
from models.loss import GANLoss
from apex import amp
import torch.nn.functional as F

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

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)

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
                self.l_fea_w_decay = train_opt['feature_weight_decay']
                self.l_fea_w_decay_steps = train_opt['feature_weight_decay_steps']
                self.l_fea_w_minimum = train_opt['feature_weight_minimum']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    pass  # do not need to use DistributedDataParallel for netF
                else:
                    self.netF = DataParallel(self.netF)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
            self.D_noise_theta = train_opt['D_noise_theta_init'] if train_opt['D_noise_theta_init'] else 0
            self.D_noise_final = train_opt['D_noise_final_it'] if train_opt['D_noise_final_it'] else 0

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
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
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

            # AMP
            [self.netG, self.netD], [self.optimizer_G, self.optimizer_D] = \
                amp.initialize([self.netG, self.netD], [self.optimizer_G, self.optimizer_D], opt_level=self.amp_level, num_losses=3)

            # DataParallel
            if opt['dist']:
                self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            else:
                self.netG = DataParallel(self.netG)
            if self.is_train:
                if opt['dist']:
                    self.netD = DistributedDataParallel(self.netD,
                                                        device_ids=[torch.cuda.current_device()])
                else:
                    self.netD = DataParallel(self.netD)
                self.netG.train()
                self.netD.train()

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed

    def feed_data(self, data, need_GT=True):
        self.var_L = torch.chunk(data['LQ'], chunks=self.mega_batch_factor, dim=0)  # LQ
        if need_GT:
            self.var_H = [t.to(self.device) for t in torch.chunk(data['GT'], chunks=self.mega_batch_factor, dim=0)]
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = [t.to(self.device) for t in torch.chunk(input_ref, chunks=self.mega_batch_factor, dim=0)]
            self.pix = [t.to(self.device) for t in torch.chunk(data['PIX'], chunks=self.mega_batch_factor, dim=0)]

    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        if step > self.D_init_iters:
            self.optimizer_G.zero_grad()

        # Turning off G-grad is required to enable mega-batching and D_update_ratio to work together for some reason.
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            for p in self.netG.parameters():
                p.requires_grad = True
        else:
            for p in self.netG.parameters():
                p.requires_grad = False

        # Calculate a standard deviation for the gaussian noise to be applied to the discriminator, termed noise-theta.
        if step >= self.D_noise_final:
            noise_theta = 0
        else:
            noise_theta = self.D_noise_theta * (self.D_noise_final - step) / self.D_noise_final

        self.fake_GenOut = []
        var_ref_skips = []
        for var_L, var_H, var_ref, pix in zip(self.var_L, self.var_H, self.var_ref, self.pix):
            fake_GenOut = self.netG(var_L)

            # Extract the image output. For generators that output skip-through connections, the master output is always
            # the first element of the tuple.
            if isinstance(fake_GenOut, tuple):
                gen_img = fake_GenOut[0]
                # TODO: Fix this.
                self.fake_GenOut.append((fake_GenOut[0].detach(),
                                         fake_GenOut[1].detach(),
                                         fake_GenOut[2].detach()))
                var_ref = (var_ref,) + self.create_artificial_skips(var_H)
            else:
                gen_img = fake_GenOut
                self.fake_GenOut.append(fake_GenOut.detach())
            var_ref_skips.append(var_ref)

            l_g_total = 0
            if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                if self.cri_pix:  # pixel loss
                    l_g_pix = self.l_pix_w * self.cri_pix(gen_img, pix)
                    l_g_total += l_g_pix
                if self.cri_fea:  # feature loss
                    real_fea = self.netF(pix).detach()
                    fake_fea = self.netF(gen_img)
                    l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                    l_g_total += l_g_fea

                    # Decay the influence of the feature loss. As the model trains, the GAN will play a stronger role
                    # in the resultant image.
                    if step % self.l_fea_w_decay_steps == 0:
                        self.l_fea_w = max(self.l_fea_w_minimum, self.l_fea_w * self.l_fea_w_decay)

                if self.opt['train']['gan_type'] == 'gan':
                    pred_g_fake = self.netD(fake_GenOut)
                    l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
                elif self.opt['train']['gan_type'] == 'ragan':
                    pred_d_real = self.netD(var_ref).detach()
                    pred_g_fake = self.netD(fake_GenOut)
                    l_g_gan = self.l_gan_w * (
                        self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                        self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                l_g_total += l_g_gan

                # Scale the loss down by the batch factor.
                l_g_total = l_g_total / self.mega_batch_factor

                with amp.scale_loss(l_g_total, self.optimizer_G, loss_id=0) as l_g_total_scaled:
                    l_g_total_scaled.backward()
        self.optimizer_G.step()

        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        noise = torch.randn_like(var_ref[0]) * noise_theta
        noise.to(self.device)
        self.optimizer_D.zero_grad()
        for var_L, var_H, var_ref, pix, fake_H in zip(self.var_L, self.var_H, var_ref_skips, self.pix, self.fake_GenOut):
            # Apply noise to the inputs to slow discriminator convergence.
            var_ref = (var_ref[0] + noise,) + var_ref[1:]
            fake_H = (fake_H[0] + noise,) + fake_H[1:]
            if self.opt['train']['gan_type'] == 'gan':
                # need to forward and backward separately, since batch norm statistics differ
                # real
                pred_d_real = self.netD(var_ref)
                l_d_real = self.cri_gan(pred_d_real, True) / self.mega_batch_factor
                with amp.scale_loss(l_d_real, self.optimizer_D, loss_id=2) as l_d_real_scaled:
                    l_d_real_scaled.backward()
                # fake
                pred_d_fake = self.netD(fake_H)
                l_d_fake = self.cri_gan(pred_d_fake, False) / self.mega_batch_factor
                with amp.scale_loss(l_d_fake, self.optimizer_D, loss_id=1) as l_d_fake_scaled:
                    l_d_fake_scaled.backward()
            elif self.opt['train']['gan_type'] == 'ragan':
                # pred_d_real = self.netD(var_ref)
                # pred_d_fake = self.netD(fake_H.detach())  # detach to avoid BP to G
                # l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                # l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
                # l_d_total = (l_d_real + l_d_fake) / 2
                # l_d_total.backward()
                pred_d_fake = self.netD(fake_H).detach()
                pred_d_real = self.netD(var_ref)
                l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True) * 0.5 / self.mega_batch_factor
                with amp.scale_loss(l_d_real, self.optimizer_D, loss_id=2) as l_d_real_scaled:
                    l_d_real_scaled.backward()
                pred_d_fake = self.netD(fake_H)
                l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real.detach()), False) * 0.5 / self.mega_batch_factor
                with amp.scale_loss(l_d_fake, self.optimizer_D, loss_id=1) as l_d_fake_scaled:
                    l_d_fake_scaled.backward()
        self.optimizer_D.step()

        # Log sample images from first microbatch.
        if step % 50 == 0:
            os.makedirs("temp/hr", exist_ok=True)
            os.makedirs("temp/lr", exist_ok=True)
            os.makedirs("temp/gen", exist_ok=True)
            os.makedirs("temp/pix", exist_ok=True)
            multi_gen = False
            if isinstance(self.fake_GenOut[0], tuple):
                os.makedirs("temp/genlr", exist_ok=True)
                os.makedirs("temp/genmr", exist_ok=True)
                os.makedirs("temp/ref", exist_ok=True)
                multi_gen = True
            for i in range(self.mega_batch_factor):
                utils.save_image(self.var_H[i].cpu().detach(), os.path.join("temp/hr", "%05i_%02i.png" % (step, i)))
                utils.save_image(self.var_L[i].cpu().detach(), os.path.join("temp/lr", "%05i_%02i.png" % (step, i)))
                utils.save_image(self.pix[i].cpu().detach(), os.path.join("temp/pix", "%05i_%02i.png" % (step, i)))
                if multi_gen:
                    utils.save_image(self.fake_GenOut[i][0].cpu().detach(), os.path.join("temp/gen", "%05i_%02i.png" % (step, i)))
                    utils.save_image(self.fake_GenOut[i][1].cpu().detach(), os.path.join("temp/genmr", "%05i_%02i.png" % (step, i)))
                    utils.save_image(self.fake_GenOut[i][2].cpu().detach(), os.path.join("temp/genlr", "%05i_%02i.png" % (step, i)))
                    utils.save_image(var_ref_skips[i][1].cpu().detach(), os.path.join("temp/ref", "med_%05i_%02i.png" % (step, i)))
                    utils.save_image(var_ref_skips[i][2].cpu().detach(), os.path.join("temp/ref", "low_%05i_%02i.png" % (step, i)))
                else:
                    utils.save_image(self.fake_GenOut[i].cpu().detach(), os.path.join("temp/gen", "%05i_%02i.png" % (step, i)))

        # set log TODO(handle mega-batches?)
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.add_log_entry('l_g_pix', l_g_pix.item())
            if self.cri_fea:
                self.add_log_entry('feature_weight', self.l_fea_w)
                self.add_log_entry('l_g_fea', l_g_fea.item())
            self.add_log_entry('l_g_gan', l_g_gan.item())
            self.add_log_entry('l_g_total', l_g_total.item() * self.mega_batch_factor)
        self.add_log_entry('l_d_real', l_d_real.item() * self.mega_batch_factor)
        self.add_log_entry('l_d_fake', l_d_fake.item() * self.mega_batch_factor)
        self.add_log_entry('D_fake', torch.mean(pred_d_fake.detach()))
        self.add_log_entry('noise_theta', noise_theta)

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


    def create_artificial_skips(self, truth_img):
        med_skip = F.interpolate(truth_img, scale_factor=.5)
        lo_skip = F.interpolate(truth_img, scale_factor=.25)
        return med_skip, lo_skip

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_GenOut = [self.netG(self.var_L[0])]
        self.netG.train()

    # Fetches a summary of the log.
    def get_current_log(self):
        return_log = {}
        for k in self.log_dict.keys():
            if not isinstance(self.log_dict[k], list):
                continue
            return_log[k] = sum(self.log_dict[k]) / len(self.log_dict[k])
        return return_log

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L[0].detach()[0].float().cpu()
        gen_batch = self.fake_GenOut[0]
        if isinstance(gen_batch, tuple):
            gen_batch = gen_batch[0]
        out_dict['rlt'] = gen_batch.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H[0].detach()[0].float().cpu()
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

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
