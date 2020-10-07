from models.steps.losses import ConfigurableLoss, GANLoss, extract_params_from_state
from models.layers.resample2d_package.resample2d import Resample2d
from models.steps.recurrent import RecurrentController
from models.steps.injectors import Injector
import torch
import os
import os.path as osp
import torchvision

def create_teco_loss(opt, env):
    type = opt['type']
    if type == 'teco_generator_gan':
        return TecoGanGeneratorLoss(opt, env)
    elif type == 'teco_discriminator_gan':
        return TecoGanDiscriminatorLoss(opt, env)
    elif type == "teco_pingpong":
        return PingPongLoss(opt, env)
    return None

def create_teco_discriminator_sextuplet(input_list, index, flow_gen, resampler):
    triplet = input_list[index:index+3]
    first_flow = flow_gen(triplet[0], triplet[1])
    last_flow = flow_gen(triplet[2], triplet[1])
    flow_triplet = [resampler(triplet[0], first_flow), triplet[1], resampler(triplet[2], last_flow)]
    return torch.cat(triplet + flow_triplet, dim=1)


# Uses a generator to synthesize a sequence of images from [in] and injects the results into a list [out]
# Images are fed in sequentially forward and back, resulting in len([out])=2*len([in])-1 (last element is not repeated).
# All computation is done with torch.no_grad().
class RecurrentImageGeneratorSequenceInjector(Injector):
    def __init__(self, opt, env):
        super(RecurrentImageGeneratorSequenceInjector, self).__init__(opt, env)
        self.flow = opt['flow_network']
        self.resample = Resample2d()

    def forward(self, state):
        gen = self.env['generators'][self.opt['generator']]
        flow = self.env['generators'][self.flow]
        results = []
        recurrent_input = torch.zeros_like(state[self.input][0])

        # Go forward in the sequence first.
        first_step = True
        for input in state[self.input]:
            if first_step:
                first_step = False
            else:
                flowfield = flow(recurrent_input, input)
                recurrent_input = self.resample(recurrent_input, flowfield)
            recurrent_input = gen(input, recurrent_input)
            results.append(recurrent_input)
            recurrent_input = self.flow()

        # Now go backwards, skipping the last element (it's already stored in recurrent_input)
        it = reversed(range(len(results) - 1))
        for i in it:
            flowfield = flow(recurrent_input, results[i])
            recurrent_input = self.resample(recurrent_input, flowfield)
            recurrent_input = gen(results[i], recurrent_input)
            results.append(recurrent_input)

        return {self.output: results}


# This is the temporal discriminator loss from TecoGAN.
#
# It has a strict contact for 'real' and 'fake' inputs:
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
class TecoGanDiscriminatorLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(TecoGanDiscriminatorLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = GANLoss(opt['gan_type'], 1.0, 0.0).to(env['device'])
        self.noise = None if 'noise' not in opt.keys() else opt['noise']
        self.image_flow_generator = opt['image_flow_generator']
        self.resampler = Resample2d()

    def forward(self, net, state):
        self.metrics = []
        flow_gen = self.env['generators'][self.image_flow_generator]
        real = state[self.opt['real']]
        fake = state[self.opt['fake']]
        l_total = 0
        for i in range(len(real) - 2):
            real_sext = create_teco_discriminator_sextuplet(real, i, flow_gen, self.resampler)
            fake_sext = create_teco_discriminator_sextuplet(fake, i, flow_gen, self.resampler)

            d_real = net(real_sext)
            d_fake = net(fake_sext)

            if self.opt['gan_type'] in ['gan', 'pixgan']:
                self.metrics.append(("d_fake", torch.mean(d_fake)))
                self.metrics.append(("d_real", torch.mean(d_real)))
                l_real = self.criterion(d_real, True)
                l_fake = self.criterion(d_fake, False)
                l_total += l_real + l_fake
            elif self.opt['gan_type'] == 'ragan':
                d_fake_diff = d_fake - torch.mean(d_real)
                self.metrics.append(("d_fake_diff", torch.mean(d_fake_diff)))
                l_total += (self.criterion(d_real - torch.mean(d_fake), True) +
                           self.criterion(d_fake_diff, False))
            else:
                raise NotImplementedError
        return l_total


class TecoGanGeneratorLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(TecoGanGeneratorLoss, self).__init__(opt, env)
        self.criterion = GANLoss(opt['gan_type'], 1.0, 0.0).to(env['device'])
        # TecoGAN parameters
        self.image_flow_generator = opt['image_flow_generator']
        self.resampler = Resample2d()

    def forward(self, _, state):
        flow_gen = self.env['generators'][self.image_flow_generator]
        real = state[self.opt['real']]
        fake = state[self.opt['fake']]
        l_total = 0
        for i in range(len(real) - 2):
            real_sext = create_teco_discriminator_sextuplet(real, i, flow_gen, self.resampler)
            fake_sext = create_teco_discriminator_sextuplet(fake, i, flow_gen, self.resampler)
            d_fake = net(fake_sext)

            if self.env['step'] % 100 == 0:
                self.produce_teco_visual_debugs(fake_sext, 'fake', i)
                self.produce_teco_visual_debugs(real_sext, 'real', i)

            if self.opt['gan_type'] in ['gan', 'pixgan']:
                self.metrics.append(("d_fake", torch.mean(d_fake)))
                l_fake = self.criterion(d_fake, True)
                l_total += l_fake
            elif self.opt['gan_type'] == 'ragan':
                d_real = net(real_sext)
                d_fake_diff = d_fake - torch.mean(d_real)
                self.metrics.append(("d_fake_diff", torch.mean(d_fake_diff)))
                l_total += (self.criterion(d_real - torch.mean(d_fake), False) +
                           self.criterion(d_fake_diff, True))
            else:
                raise NotImplementedError

        return l_total

    def produce_teco_visual_debugs(self, sext, lbl, it):
        base_path = osp.join(self.env['base_path'], "visual_dbg", "teco_sext", str(self.env['step']), lbl)
        os.makedirs(base_path, exist_ok=True)
        lbls = ['first', 'second', 'third', 'first_flow', 'second_flow', 'third_flow']
        for i in range(6):
            torchvision.utils.save_image(sext[:, i*3:(i+1)*3-1, :, :], osp.join(base_path, "%s_%s.png" % (lbls[i], it)))


# This loss doesn't have a real entry - only fakes are used.
class PingPongLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(PingPongLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = GANLoss(opt['gan_type'], 1.0, 0.0).to(env['device'])

    def forward(self, _, state):
        fake = state[self.opt['fake']]
        l_total = 0
        for i in range((len(fake) - 1) / 2):
            early = fake[i]
            late = fake[-i]
            l_total += self.criterion(early, late)

        if self.env['step'] % 100 == 0:
            self.produce_teco_visual_debugs(fake)

        return l_total

    def produce_teco_visual_debugs(self, imglist):
        base_path = osp.join(self.env['base_path'], "visual_dbg", "teco_pingpong", str(self.env['step']))
        os.makedirs(base_path, exist_ok=True)
        assert isinstance(imglist, list)
        for i, img in enumerate(imglist):
            torchvision.utils.save_image(img, osp.join(base_path, "%s.png" % (i, )))