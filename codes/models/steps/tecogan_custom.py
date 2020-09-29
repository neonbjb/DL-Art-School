from models.steps.losses import ConfigurableLoss, GANLoss, extract_params_from_state
from models.layers.resample2d_package.resample2d import Resample2d
from models.steps.recurrent import RecurrentController
from models.steps.injectors import Injector
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


# Controller class that schedules the recurring inputs of tecogan
class TecoGanController(RecurrentController):
    def __init__(self, opt, env):
        super(TecoGanController, self).__init__(opt, env)
        self.sequence_len = opt['teco_sequence_length']

    def get_next_step(self, state, recurrent_state):
        # The first stage feeds the LR input into both generator inputs.
        if recurrent_state is None:
            return {
                '_gen_lr_input_index': 0,
                '_teco_recurrent_counter': 0
                '_teco_stage': 0
            }
        # The second stage is truly recurrent, but needs its own stage counter because the temporal discriminator
        # cannot come online yet.
        elif recurrent_state['_teco_recurrent_counter'] == 1:
            return {
                '_gen_lr_input_index': 1,
                '_teco_stage': 1,
                '_teco_recurrent_counter': recurrent_state['_teco_recurrent_counter'] + 1
            }
        # The third stage is truly recurrent through the end of the sequence.
        elif recurrent_state['_teco_recurrent_counter'] < self.sequence_len:
            return {
                '_gen_lr_input_index': recurrent_state['_gen_lr_input_index'] + 1,
                '_teco_stage': 2,
                '_teco_recurrent_counter': recurrent_state['_teco_recurrent_counter'] + 1
            }
        # The fourth stage regresses backwards through the sequence.
        elif recurrent_state['_teco_recurrent_counter'] < self.sequence_len * 2 - 1:
            return {
                '_gen_lr_input_index': self.sequence_len - recurrent_state['teco_recurrent_counter'] - 1,
                '_teco_stage': 3,
                '_teco_recurrent_counter': recurrent_state['_teco_recurrent_counter'] + 1
            }
        else:
            return None


# Uses a generator to synthesize a sequence of images from [in] and injects the results into a list [out]
# Images are fed in sequentially forward and back, resulting in len([out])=2*len([in])-1 (last element is not repeated).
# All computation is done with torch.no_grad().
class RecurrentImageGeneratorSequenceInjector(Injector):
    def __init__(self, opt, env):
        super(RecurrentImageGeneratorSequenceInjector, self).__init__(opt, env)

    def forward(self, state):
        gen = self.env['generators'][self.opt['generator']]
        results = []
        with torch.no_grad():
            recurrent_input = torch.zeros_like(state[self.input][0])
            # Go forward in the sequence first.
            for input in state[self.input]:
                recurrent_input = gen(input, recurrent_input)
                results.append(recurrent_input)

            # Now go backwards, skipping the last element (it's already stored in recurrent_input)
            it = reversed(range(len(results) - 1))
            for i in it:
                recurrent_input = gen(results[i], recurrent_input)
                results.append(recurrent_input)

            new_state = {self.output: results}
        return new_state


class ImageFlowInjector(Injector):
    def __init__(self, opt, env):
        # Requires building this custom cuda kernel. Only require it if explicitly needed.
        from models.networks.layers.resample2d_package.resample2d import Resample2d
        super(ImageFlowInjector, self).__init__(opt, env)
        self.resample = Resample2d()

    def forward(self, state):
        return self.resample(state[self.opt['in']], state[self.opt['flow']])


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