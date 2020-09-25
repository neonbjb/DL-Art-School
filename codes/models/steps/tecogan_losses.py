from models.steps.losses import ConfigurableLoss, GANLoss, extract_params_from_state
from models.layers.resample2d_package.resample2d import Resample2d
import torch
from apex import amp


def create_teco_discriminator_sextuplet(input_list, index, flow_gen, resampler, detach=True):
    triplet = input_list[index:index+3]
    first_flow = flow_gen(triplet[1], triplet[0])
    last_flow = flow_gen(triplet[1], triplet[2])
    if detach:
        first_flow = first_flow.detach()
        last_flow = last_flow.detach()
    flow_triplet = [resampler(triplet[0], first_flow), triplet[1], resampler(triplet[2], last_flow)]
    return torch.cat(triplet + flow_triplet, dim=1)

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
#
#   Note: All steps before 'discriminator_flow_after' do not use triplets. Instead, they use a single image repeated 6 times across the filter dimension.
class TecoGanDiscriminatorLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(TecoGanDiscriminatorLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = GANLoss(opt['gan_type'], 1.0, 0.0).to(env['device'])
        self.discriminator_flow_after = opt['discriminator_flow_after']
        self.image_flow_generator = opt['image_flow_generator']
        self.resampler = Resample2d()

    def forward(self, net, state):
        self.metrics = []
        flow_gen = self.env['generators'][self.image_flow_generator]
        real = state[self.opt['real']]
        fake = state[self.opt['fake']]
        backwards_count = range(len(real)-2)
        for i in range(len(real) - 2):
            if self.env['']
            real_sext = create_teco_discriminator_sextuplet(real, i, flow_gen, self.resampler)
            fake_sext = create_teco_discriminator_sextuplet(fake, i, flow_gen, self.resampler)

            d_real = net(real_sext)
            d_fake = net(fake_sext)

            if self.opt['gan_type'] in ['gan', 'pixgan']:
                self.metrics.append(("d_fake", torch.mean(d_fake)))
                self.metrics.append(("d_real", torch.mean(d_real)))
                l_real = self.criterion(d_real, True)
                l_fake = self.criterion(d_fake, False)
                l_total = l_real + l_fake
            elif self.opt['gan_type'] == 'ragan':
                d_fake_diff = d_fake - torch.mean(d_real)
                self.metrics.append(("d_fake_diff", torch.mean(d_fake_diff)))
                l_total = (self.criterion(d_real - torch.mean(d_fake), True) +
                           self.criterion(d_fake_diff, False))
            else:
                raise NotImplementedError

            l_total = l_total / backwards_count
            if self.env['amp']:
                with amp.scale_loss(l_total, self.env['current_step_optimizers'][0], self.env['amp_loss_id']) as loss:
                    loss.backward()
            else:
                l_total.backward()