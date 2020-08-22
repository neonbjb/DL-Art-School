import logging
import os
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision.utils as utils
from apex import amp
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.base_model import BaseModel
from models.steps.steps import ConfigurableStep

logger = logging.getLogger('base')


class ExtensibleTrainer(BaseModel):
    def __init__(self, opt):
        super(ExtensibleTrainer, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        self.mega_batch_factor = 1

        # env is used as a global state to store things that subcomponents might need.
        env = {'device': self.device,
               'rank': self.rank,
               'opt': opt}

        self.netsG = {}
        self.netsD = {}
        self.networks = []
        for name, net in opt['networks'].items():
            if net['type'] == 'generator':
                new_net = networks.define_G(net, None, opt['scale']).to(self.device)
                self.netsG[name] = new_net
            elif net['type'] == 'discriminator':
                new_net = networks.define_D_net(net, opt['datasets']['train']['target_size']).to(self.device)
                self.netsD[name] = new_net
            else:
                raise NotImplementedError("Can only handle generators and discriminators")
            self.networks.append(new_net)

        if self.is_train:
            self.mega_batch_factor = train_opt['mega_batch_factor']
            if self.mega_batch_factor is None:
                self.mega_batch_factor = 1

            # Initialize amp.
            amp_nets, amp_opts = amp.initialize(self.networks, self.optimizers, opt_level=opt['amp_opt_level'], num_losses=len(opt['steps']))
            # self.networks is stored unwrapped. It should never be used for forward() or backward() passes, instead use
            # self.netG and self.netD for that.
            self.networks = amp_nets

            # DataParallel
            dnets = []
            for anet in amp_nets:
                if opt['dist']:
                    dnet = DistributedDataParallel(anet,
                                                   device_ids=[torch.cuda.current_device()],
                                                   find_unused_parameters=True)
                else:
                    dnet = DataParallel(anet)
                if self.is_train:
                    dnet.train()
                else:
                    dnet.eval()
                dnets.append(dnet)

            # Backpush the wrapped networks into the network dicts..
            found = 0
            for dnet in dnets:
                for net_dict in [self.netsD, self.netsG]:
                    for k, v in net_dict.items():
                        if v == dnet.module:
                            net_dict[k] = dnet
                            found += 1
            assert found == len(self.networks)

            env['generators'] = self.netsG
            env['discriminators'] = self.netsD

            # Initialize the training steps
            self.steps = []
            for step_name, step in opt['steps'].items():
                step = ConfigurableStep(step, env)
                self.steps.append(step)
                self.optimizers.extend(step.get_optimizers())

            # Find the optimizers that are using the default scheduler, then build them.
            def_opt = []
            for s in self.steps:
                def_opt.extend(s.get_optimizers_with_default_scheduler())
            lr_scheduler.get_scheduler_for_name(train_opt['default_lr_scheme'], def_opt, train_opt)

        self.print_network()  # print network
        self.load()  # load G and D if needed

        # Setting this to false triggers SRGAN to call the models update_model() function on the first iteration.
        self.updated = True

    def feed_data(self, data):
        self.var_L = torch.chunk(corrupted_L, chunks=self.mega_batch_factor, dim=0)
        self.var_H = [t.to(self.device) for t in torch.chunk(data['GT'], chunks=self.mega_batch_factor, dim=0)]
        input_ref = data['ref'] if 'ref' in data else data['GT']
        self.var_ref = [t.to(self.device) for t in torch.chunk(input_ref, chunks=self.mega_batch_factor, dim=0)]

    def optimize_parameters(self, step):
        # Some models need to make parametric adjustments per-step. Do that here.
        for net in self.networks.values():
            if hasattr(net, "update_for_step"):
                net.update_for_step(step, os.path.join(self.opt['path']['models'], ".."))

        # Iterate through the steps, performing them one at a time.
        state = {'lq': self.var_L, 'hq': self.var_H, 'ref': self.var_ref}
        for step_num, s in enumerate(self.steps):
            # Only set requires_grad=True for the network being trained.
            nets_to_train = s.get_networks_trained()
            for name, net in self.networks.items():
                net_enabled = name in nets_to_train
                for p in self.netsG.parameters():
                    if p.dtype != torch.int64 and p.dtype != torch.bool:
                        p.requires_grad = net_enabled
                    else:
                        p.requires_grad = False

            # Now do a forward and backward pass for each gradient accumulation step.
            new_states = {}
            for m in range(self.mega_batch_factor):
                ns = s.do_forward_backward(state, m, step_num)
                for k, v in ns.items():
                    if k not in new_states.keys():
                        new_states[k] = [v.detach()]
                    else:
                        new_states[k].append(v.detach())

            # Push the detached new state tensors into the state map for use with the next step.
            for k, v in new_states.items():
                # Overwriting existing state keys is not supported.
                assert k not in state.keys()
                state[k] = v

            # And finally perform optimization.
            s.do_step()



        # G
        for p in self.netsD.parameters():
            p.requires_grad = False
        if self.spsr_enabled:
            for p in self.netD_grad.parameters():
                p.requires_grad = False

        self.swapout_D(step)
        self.swapout_G(step)

        # Turning off G-grad is required to enable mega-batching and D_update_ratio to work together for some reason.
        if step % self.D_update_ratio == 0 and step >= self.D_init_iters:
            if self.spsr_enabled and self.branch_pretrain and step < self.branch_init_iters:
                for k, v in self.netsG.named_parameters():
                    if v.dtype != torch.int64 and v.dtype != torch.bool:
                        v.requires_grad = '_branch_pretrain' in k
            else:
                for p in self.netsG.parameters():
                    if p.dtype != torch.int64 and p.dtype != torch.bool:
                        p.requires_grad = True
        else:
            for p in self.netsG.parameters():
                p.requires_grad = False

        # Calculate a standard deviation for the gaussian noise to be applied to the discriminator, termed noise-theta.
        if self.D_noise_final == 0:
            noise_theta = 0
        else:
            noise_theta = (self.D_noise_theta - self.D_noise_theta_floor) * (self.D_noise_final - min(step, self.D_noise_final)) / self.D_noise_final + self.D_noise_theta_floor

        if _profile:
            print("Misc setup %f" % (time() - _t,))
            _t = time()

        if step >= self.D_init_iters:
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
                fake_H_branch, fake_GenOut, grad_LR = self.netsG(var_L)
                fea_GenOut = fake_GenOut
                self.spsr_grad_GenOut.append(fake_H_branch)
                # Get image gradients for later use.
                fake_H_grad = self.get_grad_nopadding(fake_GenOut)
            else:
                if random.random() > self.gan_lq_img_use_prob:
                    fea_GenOut, fake_GenOut = self.netsG(var_L)
                    using_gan_img = False
                else:
                    fea_GenOut, fake_GenOut = self.netsG(var_LGAN)
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
                            pred_g_fake = self.netsD(fake_GenOut, var_L)
                        else:
                            pred_g_fake = self.netsD(fake_GenOut)
                        l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
                    elif self.opt['train']['gan_type'] == 'ragan':
                        pred_d_real = self.netsD(var_ref).detach()
                        pred_g_fake = self.netsD(fake_GenOut)
                        l_g_gan = self.l_gan_w * (
                            self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                            self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                    l_g_gan_log = l_g_gan / self.l_gan_w
                    l_g_total += l_g_gan

                if self.spsr_enabled and self.cri_grad_gan:
                    if self.opt['train']['gan_type'] == 'crossgan':
                        pred_g_fake_grad = self.netsD(fake_H_grad, var_L)
                    else:
                        pred_g_fake_grad = self.netsD(fake_H_grad)
                    pred_g_fake_grad_branch = self.netD_grad(fake_H_branch)
                    if self.opt['train']['gan_type'] in ['gan', 'pixgan', 'pixgan_fea', 'crossgan']:
                        l_g_gan_grad = self.l_gan_grad_w * self.cri_grad_gan(pred_g_fake_grad, True)
                        l_g_gan_grad_branch = self.l_gan_grad_w * self.cri_grad_gan(pred_g_fake_grad_branch, True)
                    elif self.opt['train']['gan_type'] == 'ragan':
                        pred_g_real_grad = self.netD_grad(self.get_grad_nopadding(var_ref)).detach()
                        l_g_gan_grad = self.l_gan_w * (
                            self.cri_gan(pred_g_real_grad - torch.mean(pred_g_fake_grad), False) +
                            self.cri_gan(pred_g_fake_grad - torch.mean(pred_g_real_grad), True)) / 2
                        l_g_gan_grad_branch = self.l_gan_w * (
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
            for p in self.netsD.parameters():
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
                            _, fake_H, _ = self.netsG(var_LGAN)
                        else:
                            _, fake_H = self.netsG(var_LGAN)
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
                    _, fea_real = self.netsD(var_ref, output_feature_vector=True)
                    actual_fea = self.netF(var_ref)
                    l_d_fea_real = self.cri_fea(fea_real, actual_fea) * disc_fea_scale / self.mega_batch_factor
                    _, fea_fake = self.netsD(fake_H, output_feature_vector=True)
                    actual_fea = self.netF(fake_H)
                    l_d_fea_fake = self.cri_fea(fea_fake, actual_fea) * disc_fea_scale / self.mega_batch_factor
                if self.opt['train']['gan_type'] == 'crossgan':
                    # need to forward and backward separately, since batch norm statistics differ
                    # real
                    pred_d_real = self.netsD(var_ref, var_L)
                    l_d_real = self.cri_gan(pred_d_real, True)
                    l_d_real_log = l_d_real
                    # fake
                    pred_d_fake = self.netsD(fake_H, var_L)
                    l_d_fake = self.cri_gan(pred_d_fake, False)
                    l_d_fake_log = l_d_fake
                    # mismatched
                    mismatched_L = torch.roll(var_L, shifts=1, dims=0)
                    pred_d_real_mismatched = self.netsD(var_ref, mismatched_L)
                    pred_d_fake_mismatched = self.netsD(fake_H, mismatched_L)
                    l_d_mismatched = (self.cri_gan(pred_d_real_mismatched, False) + self.cri_gan(pred_d_fake_mismatched, False)) / 2

                    l_d_total = (l_d_real + l_d_fake + l_d_mismatched) / 3
                    l_d_total = l_d_total / self.mega_batch_factor
                    with amp.scale_loss(l_d_total, self.optimizer_D, loss_id=1) as l_d_total_scaled:
                        l_d_total_scaled.backward()
                elif self.opt['train']['gan_type'] == 'gan':
                    # real
                    pred_d_real = self.netsD(var_ref)
                    l_d_real = self.cri_gan(pred_d_real, True) / self.mega_batch_factor
                    l_d_real_log = l_d_real * self.mega_batch_factor
                    # fake
                    pred_d_fake = self.netsD(fake_H)
                    l_d_fake = self.cri_gan(pred_d_fake, False) / self.mega_batch_factor
                    l_d_fake_log = l_d_fake * self.mega_batch_factor

                    l_d_total = (l_d_real + l_d_fake) / 2
                    with amp.scale_loss(l_d_total, self.optimizer_D, loss_id=1) as l_d_total_scaled:
                        l_d_total_scaled.backward()
                elif 'pixgan' in self.opt['train']['gan_type']:
                    pixdisc_channels, pixdisc_output_reduction = self.netsD.module.pixgan_parameters()
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
                    pred_d_real = self.netsD(var_ref)
                    l_d_real = self.cri_gan(pred_d_real, real) / self.mega_batch_factor
                    l_d_real_log = l_d_real * self.mega_batch_factor
                    l_d_real += l_d_fea_real
                    # fake
                    pred_d_fake = self.netsD(fake_H)
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
                    pred_d_fake = self.netsD(fake_H)
                    pred_d_real = self.netsD(var_ref)
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
                for var_ref, fake_H, fake_H_grad_branch in zip(var_ref_skips, self.fake_H, self.spsr_grad_GenOut):
                    fake_H_grad = self.get_grad_nopadding(fake_H).detach()
                    var_ref_grad = self.get_grad_nopadding(var_ref)
                    pred_d_real_grad = self.netD_grad(var_ref_grad)
                    pred_d_fake_grad = self.netD_grad(fake_H_grad)   # Tensor already detached above.
                    # var_ref and fake_H already has noise added to it. We **must** add noise to fake_H_grad_branch too.
                    fake_H_grad_branch = fake_H_grad_branch.detach() + noise
                    pred_d_fake_grad_branch = self.netD_grad(fake_H_grad_branch)
                    if self.opt['train']['gan_type'] == 'gan':
                        l_d_real_grad = self.cri_gan(pred_d_real_grad, True)
                        l_d_fake_grad = (self.cri_gan(pred_d_fake_grad, False) + self.cri_gan(pred_d_fake_grad_branch, False)) / 2
                    elif self.opt['train']['gan_type'] == 'crossgan':
                        assert False
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
                    self.add_log_entry('l_g_gan_grad', l_g_gan_grad.detach().item())
                    self.add_log_entry('l_g_gan_grad_branch', l_g_gan_grad_branch.detach().item())
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

    def compute_fea_loss(self, real, fake):
        with torch.no_grad():
            real = real.unsqueeze(dim=0).to(self.device)
            fake = fake.unsqueeze(dim=0).to(self.device)
            real_fea = self.netF(real).detach()
            fake_fea = self.netF(fake)
            return self.cri_fea(fake_fea, real_fea).item()

    def test(self):
        self.netsG.eval()
        with torch.no_grad():
            if self.spsr_enabled:
                self.fake_H_branch = []
                self.fake_GenOut = []
                self.grad_LR = []
                fake_H_branch, fake_GenOut, grad_LR = self.netsG(self.var_L[0])
                self.fake_H_branch.append(fake_H_branch)
                self.fake_GenOut.append(fake_GenOut)
                self.grad_LR.append(grad_LR)
            else:
                self.fake_GenOut = [self.netsG(self.var_L[0])]
        self.netsG.train()

    # Fetches a summary of the log.
    def get_current_log(self, step):
        return_log = {}
        for k in self.log_dict.keys():
            if not isinstance(self.log_dict[k], list):
                continue
            return_log[k] = sum(self.log_dict[k]) / len(self.log_dict[k])

        # Some generators can do their own metric logging.
        if hasattr(self.netsG.module, "get_debug_values"):
            return_log.update(self.netsG.module.get_debug_values(step))
        if hasattr(self.netsD.module, "get_debug_values"):
            return_log.update(self.netsD.module.get_debug_values(step))

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
        for name, net in self.networks.items():
            s, n = self.get_network_description(net)
            net_struc_str = '{}'.format(net.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network ' + name + ' structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        for name, net in self.networks.items():
            load_path = opt['path'][name]
            if load_path is not None:
                logger.info('Loading model for %s: [%s]' % (name, load_path))
                self.load_network(load_path, net)

    def save(self, iter_step):
        for name, net in self.networks.items():
            self.save_network(net, name, iter_step)
