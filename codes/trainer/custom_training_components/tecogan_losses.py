from torch.cuda.amp import autocast

from models.image_generation.stylegan.stylegan2_lucidrains import gradient_penalty
from trainer.losses import ConfigurableLoss, GANLoss, extract_params_from_state, get_basic_criterion_for_name
from models.flownet2.networks import Resample2d
from trainer.inject import Injector
import torch
import torch.nn.functional as F
import os
import os.path as osp
import torchvision


def create_teco_loss(opt, env):
    type = opt['type']
    if type == 'teco_gan':
        return TecoGanLoss(opt, env)
    elif type == "teco_pingpong":
        return PingPongLoss(opt, env)
    return None

def create_teco_injector(opt, env):
    type = opt['type']
    if type == 'teco_recurrent_generated_sequence_injector':
        return RecurrentImageGeneratorSequenceInjector(opt, env)
    elif type == 'teco_flow_adjustment':
        return FlowAdjustment(opt, env)
    return None

def extract_inputs_index(inputs, i):
    res = []
    for input in inputs:
        if isinstance(input, torch.Tensor):
            res.append(input[:, i])
        else:
            res.append(input)
    return res

# Uses a generator to synthesize a sequence of images from [in] and injects the results into a list [out]
# Images are fed in sequentially forward and back, resulting in len([out])=2*len([in])-1 (last element is not repeated).
# All computation is done with torch.no_grad().
class RecurrentImageGeneratorSequenceInjector(Injector):
    def __init__(self, opt, env):
        super(RecurrentImageGeneratorSequenceInjector, self).__init__(opt, env)
        self.flow = opt['flow_network']
        self.input_lq_index = opt['input_lq_index'] if 'input_lq_index' in opt.keys() else 0
        self.recurrent_index = opt['recurrent_index']
        self.output_hq_index = opt['output_hq_index'] if 'output_hq_index' in opt.keys() else 0
        self.output_recurrent_index = opt['output_recurrent_index'] if 'output_recurrent_index' in opt.keys() else self.output_hq_index
        self.scale = opt['scale']
        self.resample = Resample2d()
        self.flow_key = opt['flow_input_key'] if 'flow_input_key' in opt.keys() else None
        self.first_inputs = opt['first_inputs'] if 'first_inputs' in opt.keys() else opt['in']  # Use this to specify inputs that will be used in the first teco iteration, the rest will use 'in'.
        self.do_backwards = opt['do_backwards'] if 'do_backwards' in opt.keys() else True
        self.hq_recurrent = opt['hq_recurrent'] if 'hq_recurrent' in opt.keys() else False  # When True, recurrent_index is not touched for the first iteration, allowing you to specify what is fed in. When False, zeros are fed into the recurrent index.
        self.hq_batched_output_key = opt['hq_batched_key'] if 'hq_batched_key' in opt.keys() else None

    def forward(self, state):
        gen = self.env['generators'][self.opt['generator']]
        flow = self.env['generators'][self.flow]
        first_inputs = extract_params_from_state(self.first_inputs, state)
        inputs = extract_params_from_state(self.input, state)
        if not isinstance(inputs, list):
            inputs = [inputs]

        if not isinstance(self.output, list):
            self.output = [self.output]
        results = {}
        for out_key in self.output:
            results[out_key] = []

        # Go forward in the sequence first.
        first_step = True
        b, f, c, h, w = inputs[self.input_lq_index].shape
        debug_index = 0
        for i in range(f):
            if first_step:
                input = extract_inputs_index(first_inputs, i)
                if self.hq_recurrent:
                    recurrent_input = input[self.recurrent_index]
                else:
                    recurrent_input = torch.zeros_like(input[self.recurrent_index])
                first_step = False
            else:
                input = extract_inputs_index(inputs, i)
                with torch.no_grad() and autocast(enabled=False):
                    if self.flow_key is not None:
                        flow_input = state[self.flow_key][:, i]
                    else:
                        flow_input = input[self.input_lq_index]
                    reduced_recurrent = F.interpolate(hq_recurrent, scale_factor=1/self.scale, mode='bicubic')
                    flow_input = torch.stack([flow_input, reduced_recurrent], dim=2).float()
                    flowfield = flow(flow_input)
                    if recurrent_input.shape[-1] != flow_input.shape[-1]:
                        flowfield = F.interpolate(flowfield, scale_factor=self.scale, mode='bicubic')
                    recurrent_input = self.resample(recurrent_input.float(), flowfield)
            input[self.recurrent_index] = recurrent_input
            if self.env['step'] % 50 == 0:
                if input[self.input_lq_index].shape[1] == 3:   # Only debug this if we're dealing with images.
                    self.produce_teco_visual_debugs(input[self.input_lq_index], input[self.hq_recurrent], debug_index)
                    debug_index += 1

            with autocast(enabled=self.env['opt']['fp16']):
                gen_out = gen(*input)

            if isinstance(gen_out, torch.Tensor):
                gen_out = [gen_out]
            for i, out_key in enumerate(self.output):
                results[out_key].append(gen_out[i])
            hq_recurrent = gen_out[self.output_hq_index]
            recurrent_input = gen_out[self.output_recurrent_index]

        # Now go backwards, skipping the last element (it's already stored in recurrent_input)
        if self.do_backwards:
            it = reversed(range(f - 1))
            for i in it:
                input = extract_inputs_index(inputs, i)
                with torch.no_grad():
                    with autocast(enabled=False):
                        if self.flow_key is not None:
                            flow_input = state[self.flow_key][:, i]
                        else:
                            flow_input = input[self.input_lq_index]
                        reduced_recurrent = F.interpolate(hq_recurrent, scale_factor=1/self.scale, mode='bicubic')
                        flow_input = torch.stack([flow_input, reduced_recurrent], dim=2).float()
                        flowfield = flow(flow_input)
                        if recurrent_input.shape[-1] != flow_input.shape[-1]:
                            flowfield = F.interpolate(flow(flow_input), scale_factor=self.scale, mode='bicubic')
                        recurrent_input = self.resample(recurrent_input.float(), flowfield)
                input[self.recurrent_index] = recurrent_input
                if self.env['step'] % 50 == 0:
                    if input[self.input_lq_index].shape[1] == 3:   # Only debug this if we're dealing with images.
                        self.produce_teco_visual_debugs(input[self.input_lq_index], input[self.recurrent_index], debug_index)
                        debug_index += 1

                with autocast(enabled=self.env['opt']['fp16']):
                    gen_out = gen(*input)

                if isinstance(gen_out, torch.Tensor):
                    gen_out = [gen_out]
                for i, out_key in enumerate(self.output):
                    results[out_key].append(gen_out[i])
                hq_recurrent = gen_out[self.output_hq_index]
                recurrent_input = gen_out[self.output_recurrent_index]

        final_results = {}
        # Include 'hq_batched' here - because why not... Don't really need a separate injector for this.
        b, s, c, h, w = state['hq'].shape
        if self.hq_batched_output_key is not None:
            final_results[self.hq_batched_output_key] = state['hq'].clone().permute(1,0,2,3,4).reshape(b*s, c, h, w)
        for k, v in results.items():
            final_results[k] = torch.stack(v, dim=1)
            final_results[k + "_batched"] = torch.cat(v[:s], dim=0)  # Only include the original sequence - this output is generally used to compare against HQ.
        return final_results

    def produce_teco_visual_debugs(self, gen_input, gen_recurrent, it):
        if self.env['rank'] > 0:
            return
        base_path = osp.join(self.env['base_path'], "../../models", "visual_dbg", "teco_geninput", str(self.env['step']))
        os.makedirs(base_path, exist_ok=True)
        torchvision.utils.save_image(gen_input.float(), osp.join(base_path, "%s_img.png" % (it,)))
        torchvision.utils.save_image(gen_recurrent.float(), osp.join(base_path, "%s_recurrent.png" % (it,)))


class FlowAdjustment(Injector):
    def __init__(self, opt, env):
        super(FlowAdjustment, self).__init__(opt, env)
        self.resample = Resample2d()
        self.flow = opt['flow_network']
        self.flow_target = opt['flow_target']
        self.flowed = opt['flowed']

    def forward(self, state):
        with autocast(enabled=False):
            flow = self.env['generators'][self.flow]
            flow_target = state[self.flow_target]
            flowed = F.interpolate(state[self.flowed], size=flow_target.shape[2:], mode='bicubic')
            flow_input = torch.stack([flow_target, flowed], dim=2).float()
            flowfield = F.interpolate(flow(flow_input), size=state[self.flowed].shape[2:], mode='bicubic')
            return {self.output: self.resample(state[self.flowed], flowfield)}


def create_teco_discriminator_sextuplet(input_list, lr_imgs, scale, index, flow_gen, resampler, margin):
    # Flow is interpreted from the LR images so that the generator cannot learn to manipulate it.
    with autocast(enabled=False):
        triplet = input_list[:, index:index+3].float()
        first_flow = flow_gen(torch.stack([triplet[:,1], triplet[:,0]], dim=2))
        last_flow = flow_gen(torch.stack([triplet[:,1], triplet[:,2]], dim=2))
        flow_triplet = [resampler(triplet[:,0], first_flow),
                        triplet[:,1],
                        resampler(triplet[:,2], last_flow)]
        flow_triplet = torch.stack(flow_triplet, dim=1)
        combined = torch.cat([triplet, flow_triplet], dim=1)
        b, f, c, h, w = combined.shape
        combined = combined.view(b, 3*6, h, w)  # 3*6 is essentially an assertion here.
    # Apply margin
    return combined[:, :, margin:-margin, margin:-margin]


def create_all_discriminator_sextuplets(input_list, lr_imgs, scale, total, flow_gen, resampler, margin):
    with autocast(enabled=False):
        input_list = input_list.float()        
        # Combine everything and feed it into the flow network at once for better efficiency.
        batch_sz = input_list.shape[0]
        flux_doubles_forward = [torch.stack([input_list[:,i], input_list[:,i+1]], dim=2) for i in range(1, total+1)]
        flux_doubles_backward = [torch.stack([input_list[:,i], input_list[:,i-1]], dim=2) for i in range(1, total+1)]
        flows_forward = flow_gen(torch.cat(flux_doubles_forward, dim=0))
        flows_backward = flow_gen(torch.cat(flux_doubles_backward, dim=0))
        sexts = []
        for i in range(total):
            flow_forward = flows_forward[batch_sz*i:batch_sz*(i+1)]
            flow_backward = flows_backward[batch_sz*i:batch_sz*(i+1)]
            mid = input_list[:,i+1]
            sext = torch.stack([input_list[:,i], mid, input_list[:,i+2],
                              resampler(input_list[:,i], flow_backward),
                              mid,
                              resampler(input_list[:,i+2], flow_forward)], dim=1)
            # Apply margin
            b, f, c, h, w = sext.shape
            sext = sext.view(b, 3*6, h, w)  # f*c = 6*3
            sext = sext[:, :, margin:-margin, margin:-margin]
            sexts.append(sext)
    return torch.cat(sexts, dim=0)


# This is the temporal discriminator loss from TecoGAN.
#
# It has a strict contract for 'real' and 'fake' inputs:
#   'real' - Must be a list of arbitrary images (len>3) drawn from the dataset
#   'fake' - The output of the RecurrentImageGeneratorSequenceInjector for the same set of images.
#
# This loss does the following:
# 1) Picks an image triplet, starting with the first '3' elements in 'real' and 'fake'.
# 2) Uses the image flow generator (specified with 'image_flow_generator') to create detached flow fields for the first and last images in the above sequence.
# 3) Warps the first and last images according to the flow field.
# 4) Composes the three base image and the 2 warped images and middle image into a tensor concatenated at the filter dimension for both real and fake, resulting in a bx18xhxw shape tensor.
# 5) Feeds the catted real and fake image sets into the discriminator, computes a loss, and backward().
# 6) Repeat from (1) until all triplets from the real sequence have been exhausted.
class TecoGanLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(TecoGanLoss, self).__init__(opt, env)
        self.criterion = GANLoss(opt['gan_type'], 1.0, 0.0).to(env['device'])
        # TecoGAN parameters
        self.scale = opt['scale']
        self.lr_inputs = opt['lr_inputs']
        self.image_flow_generator = opt['image_flow_generator']
        self.resampler = Resample2d()
        self.for_generator = opt['for_generator']
        self.min_loss = opt['min_loss'] if 'min_loss' in opt.keys() else 0
        self.margin = opt['margin']  # Per the tecogan paper, the GAN loss only pays attention to an inner part of the image with the margin removed, to get rid of artifacts resulting from flow errors.
        self.ff = opt['fast_forward'] if 'fast_forward' in opt.keys() else False
        self.noise = opt['noise'] if 'noise' in opt.keys() else 0
        self.gradient_penalty = opt['gradient_penalty'] if 'gradient_penalty' in opt.keys() else False

    def forward(self, _, state):
        if self.ff:
            return self.fast_forward(state)
        else:
            return self.lowmem_forward(state)


    # Computes the discriminator loss one recursive step at a time, which has a lower memory overhead but is
    # slower.
    def lowmem_forward(self, state):
        flow_gen = self.env['generators'][self.image_flow_generator]
        real = state[self.opt['real']]
        fake = state[self.opt['fake']]
        sequence_len = real.shape[1]
        lr = state[self.opt['lr_inputs']]
        l_total = 0

        # Create a list of all the discriminator inputs, which will be reduced into the batch dim for efficient computation.
        for i in range(sequence_len - 2):
            real_sext = create_teco_discriminator_sextuplet(real, lr, self.scale, i, flow_gen, self.resampler, self.margin)
            if self.gradient_penalty:
                real_sext.requires_grad_()
            fake_sext = create_teco_discriminator_sextuplet(fake, lr, self.scale, i, flow_gen, self.resampler, self.margin)
            l_step, d_real = self.compute_loss(real_sext, fake_sext)
            if l_step > self.min_loss:
                l_total = l_total + l_step
            elif self.gradient_penalty:
                gp = gradient_penalty(real_sext, d_real)
                l_total = l_total + gp

        return l_total

    # Computes the discriminator loss by dogpiling all of the sextuplets into the batch dimension and doing one massive
    # forward() on the discriminators. High memory but faster.
    def fast_forward(self, state):
        flow_gen = self.env['generators'][self.image_flow_generator]
        real = state[self.opt['real']]
        fake = state[self.opt['fake']]
        sequence_len = real.shape[1]
        lr = state[self.opt['lr_inputs']]

        # Create a list of all the discriminator inputs, which will be reduced into the batch dim for efficient computation.
        combined_real_sext = create_all_discriminator_sextuplets(real, lr, self.scale, sequence_len - 2, flow_gen,
                                                                 self.resampler, self.margin)
        if self.gradient_penalty:
            combined_real_sext.requires_grad_()
        combined_fake_sext = create_all_discriminator_sextuplets(fake, lr, self.scale, sequence_len - 2, flow_gen,
                                                                 self.resampler, self.margin)
        l_total, d_real = self.compute_loss(combined_real_sext, combined_fake_sext)
        if l_total < self.min_loss:
            l_total = 0
        elif self.gradient_penalty:
            gp = gradient_penalty(combined_real_sext, d_real)
            l_total = l_total + gp
        return l_total

    def compute_loss(self, real_sext, fake_sext):
        fp16 = self.env['opt']['fp16']
        net = self.env['discriminators'][self.opt['discriminator']]
        if self.noise != 0:
            real_sext = real_sext + torch.rand_like(real_sext) * self.noise
            fake_sext = fake_sext + torch.rand_like(fake_sext) * self.noise
        with autocast(enabled=fp16):
            d_fake = net(fake_sext)
            d_real = net(real_sext)

        self.metrics.append(("d_fake", torch.mean(d_fake)))
        self.metrics.append(("d_real", torch.mean(d_real)))

        if self.for_generator and self.env['step'] % 50 == 0:
            self.produce_teco_visual_debugs(fake_sext, 'fake', 0)
            self.produce_teco_visual_debugs(real_sext, 'real', 0)

        if self.opt['gan_type'] in ['gan', 'pixgan']:
            l_fake = self.criterion(d_fake, self.for_generator)
            if not self.for_generator:
                l_real = self.criterion(d_real, True)
            else:
                l_real = 0
            l_step = l_fake + l_real
        elif self.opt['gan_type'] == 'ragan':
            d_fake_diff = d_fake - torch.mean(d_real)
            self.metrics.append(("d_fake_diff", torch.mean(d_fake_diff)))
            l_step = (self.criterion(d_real - torch.mean(d_fake), not self.for_generator) +
                      self.criterion(d_fake_diff, self.for_generator))
        else:
            raise NotImplementedError

        return l_step, d_real

    def produce_teco_visual_debugs(self, sext, lbl, it):
        if self.env['rank'] > 0:
            return
        base_path = osp.join(self.env['base_path'], "../../models", "visual_dbg", "teco_sext", str(self.env['step']), lbl)
        os.makedirs(base_path, exist_ok=True)
        lbls = ['img_a', 'img_b', 'img_c', 'flow_a', 'flow_b', 'flow_c']
        for i in range(6):
            torchvision.utils.save_image(sext[:, i*3:(i+1)*3, :, :].float(), osp.join(base_path, "%s_%s.png" % (it, lbls[i])))


# This loss doesn't have a real entry - only fakes are used.
class PingPongLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(PingPongLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = get_basic_criterion_for_name(opt['criterion'], env['device'])

    def forward(self, _, state):
        fake = state[self.opt['fake']]
        l_total = 0
        img_count = fake.shape[1]
        for i in range((img_count - 1) // 2):
            early = fake[:, i]
            late = fake[:, -(i+1)]
            l_total += self.criterion(early, late)
            #if self.env['step'] % 50 == 0:
            #    self.produce_teco_visual_debugs2(early, late, i)

        if self.env['step'] % 50 == 0:
            self.produce_teco_visual_debugs(fake)

        return l_total

    def produce_teco_visual_debugs(self, imglist):
        if self.env['rank'] > 0:
            return
        base_path = osp.join(self.env['base_path'], "../../models", "visual_dbg", "teco_pingpong", str(self.env['step']))
        os.makedirs(base_path, exist_ok=True)
        cnt = imglist.shape[1]
        for i in range(cnt):
            img = imglist[:, i]
            torchvision.utils.save_image(img.float(), osp.join(base_path, "%s.png" % (i, )))

    def produce_teco_visual_debugs2(self, imga, imgb, i):
        if self.env['rank'] > 0:
            return
        base_path = osp.join(self.env['base_path'], "../../models", "visual_dbg", "teco_pingpong", str(self.env['step']))
        os.makedirs(base_path, exist_ok=True)
        torchvision.utils.save_image(imga.float(), osp.join(base_path, "%s_a.png" % (i, )))
        torchvision.utils.save_image(imgb.float(), osp.join(base_path, "%s_b.png" % (i, )))

