import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from trainer.loss import GANLoss
import random
import functools
import torch.nn.functional as F

from utils.util import opt_get


def create_loss(opt_loss, env):
    type = opt_loss['type']
    if 'teco_' in type:
        from trainer.custom_training_components import create_teco_loss
        return create_teco_loss(opt_loss, env)
    elif 'stylegan2_' in type:
        from models.image_generation.stylegan import create_stylegan2_loss
        return create_stylegan2_loss(opt_loss, env)
    elif type == 'crossentropy' or type == 'cross_entropy':
        return CrossEntropy(opt_loss, env)
    elif type == 'distillation':
        return Distillation(opt_loss, env)
    elif type == 'pix':
        return PixLoss(opt_loss, env)
    elif type == 'sr_pix':
        return SrPixLoss(opt_loss, env)
    elif type == 'direct':
        return DirectLoss(opt_loss, env)
    elif type == 'feature':
        return FeatureLoss(opt_loss, env)
    elif type == 'interpreted_feature':
        return InterpretedFeatureLoss(opt_loss, env)
    elif type == 'generator_gan':
        return GeneratorGanLoss(opt_loss, env)
    elif type == 'discriminator_gan':
        return DiscriminatorGanLoss(opt_loss, env)
    elif type == 'geometric':
        return GeometricSimilarityGeneratorLoss(opt_loss, env)
    elif type == 'translational':
        return TranslationInvarianceLoss(opt_loss, env)
    elif type == 'recursive':
        return RecursiveInvarianceLoss(opt_loss, env)
    elif type == 'recurrent':
        return RecurrentLoss(opt_loss, env)
    elif type == 'for_element':
        return ForElementLoss(opt_loss, env)
    elif type == 'nv_tacotron2_loss':
        from models.audio.tts.tacotron2 import Tacotron2Loss
        return Tacotron2Loss(opt_loss, env)
    else:
        raise NotImplementedError


# Converts params to a list of tensors extracted from state. Works with list/tuple params as well as scalars.
def extract_params_from_state(params: object, state: object, root: object = True) -> object:
    if isinstance(params, list) or isinstance(params, tuple):
        p = [extract_params_from_state(r, state, False) for r in params]
    elif isinstance(params, str):
        if params == 'None':
            p = None
        else:
            p = state[params]
    else:
        p = params
    # The root return must always be a list.
    if root and not isinstance(p, list):
        p = [p]
    return p


class ConfigurableLoss(nn.Module):
    def __init__(self, opt, env):
        super(ConfigurableLoss, self).__init__()
        self.opt = opt
        self.env = env
        self.metrics = []

    # net is either a scalar network being trained or a list of networks being trained, depending on the configuration.
    def forward(self, net, state):
        raise NotImplementedError

    def is_stateful(self) -> bool:
        """
        Losses can inject into the state too. useful for when a loss computation can be used by another loss.
        if this is true, the forward pass must return (loss, new_state). If false (the default), forward() only returns
        the loss value.
        """
        return False

    def extra_metrics(self):
        return self.metrics

    def clear_metrics(self):
        self.metrics = []


def get_basic_criterion_for_name(name, device):
    if name == 'l1':
        return nn.L1Loss().to(device)
    elif name == 'l2':
        return nn.MSELoss().to(device)
    elif name == 'cosine':
        return nn.CosineEmbeddingLoss().to(device)
    else:
        raise NotImplementedError


class CrossEntropy(ConfigurableLoss):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.opt = opt
        self.subtype = opt_get(opt, ['subtype'], 'ce')
        if self.subtype == 'ce' or self.subtype == 'soft_ce':
            self.ce = nn.CrossEntropyLoss()
        elif self.subtype == 'bce':
            self.ce = nn.BCEWithLogitsLoss()
        else:
            assert False

    def forward(self, _, state):
        logits = state[self.opt['logits']]
        labels = state[self.opt['labels']]
        if self.opt['rescale']:
            labels = F.interpolate(labels.type(torch.float), size=logits.shape[2:], mode="nearest").type(torch.long)
        if 'mask' in self.opt.keys():
            mask = state[self.opt['mask']]
            if self.opt['rescale']:
                mask = F.interpolate(mask, size=logits.shape[2:], mode="nearest")
            logits = logits * mask
        if self.opt['swap_channels']:
            logits = logits.permute(0,2,3,1).contiguous()
        if self.subtype == 'bce':
            logits = logits.reshape(-1, 1)
            labels = labels.reshape(-1, 1)
        elif self.subtype == 'ce':
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            assert labels.max()+1 <= logits.shape[-1]
        elif self.subtype == 'soft_ce':
            labels = F.softmax(labels, dim=1)
            return F.cross_entropy(logits, labels)
        return self.ce(logits, labels)


class Distillation(ConfigurableLoss):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.opt = opt
        self.teacher = opt['teacher']
        self.student = opt['student']
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.temperature = opt_get(opt, ['temperature'], 1.0)

    def forward(self, _, state):
        # Current assumption is that both logits are of shape [b,C,d], b=batch,C=class_logits,d=sequence_len        
        teacher = state[self.teacher].permute(0,2,1)
        student = state[self.student].permute(0,2,1)
        
        return self.loss(input=F.log_softmax(student/self.temperature, dim=-1), target=F.softmax(teacher/self.temperature, dim=-1))

    
class PixLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(PixLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = get_basic_criterion_for_name(opt['criterion'], env['device'])
        self.real_scale = opt['real_scale'] if 'real_scale' in opt.keys() else 1
        self.real_offset = opt['real_offset'] if 'real_offset' in opt.keys() else 0
        self.report_metrics = opt['report_metrics'] if 'report_metrics' in opt.keys() else False

    def forward(self, _, state):
        real = state[self.opt['real']] * self.real_scale + float(self.real_offset)
        fake = state[self.opt['fake']]
        if self.report_metrics:
            self.metrics.append(("real_pix_mean_histogram", torch.mean(real, dim=[1,2,3]).detach()))
            self.metrics.append(("fake_pix_mean_histogram", torch.mean(fake, dim=[1,2,3]).detach()))
            self.metrics.append(("real_pix_std", torch.std(real).detach()))
            self.metrics.append(("fake_pix_std", torch.std(fake).detach()))
        return self.criterion(fake.float(), real.float())


class SrPixLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.opt = opt
        self.base_loss = opt_get(opt, ['base_loss'], .2)
        self.exp = opt_get(opt, ['exp'], 2)
        self.scale = opt['scale']

    def forward(self, _, state):
        real = state[self.opt['real']]
        fake = state[self.opt['fake']]
        l2 = (fake - real) ** 2
        self.metrics.append(("l2_loss", l2.mean()))
        # Adjust loss by prioritizing reconstruction of HF details.
        no_hf = F.interpolate(F.interpolate(real, scale_factor=1/self.scale, mode="area"), scale_factor=self.scale, mode="nearest")
        weights = (torch.abs(real - no_hf) + self.base_loss) ** self.exp
        weights = weights / weights.mean()
        loss = l2*weights
        # Preserve the intensity of the loss, just adjust the weighting.
        loss = loss*l2.mean()/loss.mean()
        return loss.mean()


# Loss defined by averaging the input tensor across all dimensions and optionally inverting it.
class DirectLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(DirectLoss, self).__init__(opt, env)
        self.opt = opt
        self.inverted = opt['inverted'] if 'inverted' in opt.keys() else False
        self.key = opt['key']
        self.anneal = opt_get(opt, ['annealing_termination_step'], 0)

    def forward(self, _, state):
        if self.inverted:
            loss = -torch.mean(state[self.key])
        else:
            loss = torch.mean(state[self.key])
        if self.anneal > 0:
            loss = loss * (1 - (self.anneal - min(self.env['step'], self.anneal)) / self.anneal)
        return loss


class FeatureLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(FeatureLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = get_basic_criterion_for_name(opt['criterion'], env['device'])
        import trainer.networks
        self.netF = trainer.networks.define_F(which_model=opt['which_model_F'],
                                              load_path=opt['load_path'] if 'load_path' in opt.keys() else None).to(self.env['device'])
        if not env['opt']['dist']:
            self.netF = torch.nn.parallel.DataParallel(self.netF, device_ids=env['opt']['gpu_ids'])

    def forward(self, _, state):
        with autocast(enabled=self.env['opt']['fp16']):
            with torch.no_grad():
                logits_real = self.netF(state[self.opt['real']])
            logits_fake = self.netF(state[self.opt['fake']])
        if self.opt['criterion'] == 'cosine':
            return self.criterion(logits_fake.float(), logits_real.float(), torch.ones(1, device=logits_fake.device))
        else:
            return self.criterion(logits_fake.float(), logits_real.float())


# Special form of feature loss which first computes the feature embedding for the truth space, then uses a second
# network which was trained to replicate that embedding on an altered input space (for example, LR or greyscale) to
# compute the embedding in the generated space. Useful for weakening the influence of the feature network in controlled
# ways.
class InterpretedFeatureLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(InterpretedFeatureLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = get_basic_criterion_for_name(opt['criterion'], env['device'])
        import trainer.networks
        self.netF_real = trainer.networks.define_F(which_model=opt['which_model_F']).to(self.env['device'])
        self.netF_gen = trainer.networks.define_F(which_model=opt['which_model_F'], load_path=opt['load_path']).to(self.env['device'])
        if not env['opt']['dist']:
            self.netF_real = torch.nn.parallel.DataParallel(self.netF_real)
            self.netF_gen = torch.nn.parallel.DataParallel(self.netF_gen)

    def forward(self, _, state):
        logits_real = self.netF_real(state[self.opt['real']])
        logits_fake = self.netF_gen(state[self.opt['fake']])
        return self.criterion(logits_fake.float(), logits_real.float())


class GeneratorGanLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(GeneratorGanLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = GANLoss(opt['gan_type'], 1.0, 0.0).to(env['device'])
        self.noise = None if 'noise' not in opt.keys() else opt['noise']
        self.detach_real = opt['detach_real'] if 'detach_real' in opt.keys() else True
        # This is a mechanism to prevent backpropagation for a GAN loss if it goes too low. This can be used to balance
        # generators and discriminators by essentially having them skip steps while their counterparts "catch up".
        self.min_loss = opt['min_loss'] if 'min_loss' in opt.keys() else 0
        if self.min_loss != 0:
            self.loss_rotating_buffer = torch.zeros(10, requires_grad=False)
            self.rb_ptr = 0
            self.losses_computed = 0

    def forward(self, _, state):
        netD = self.env['discriminators'][self.opt['discriminator']]
        real = extract_params_from_state(self.opt['real'], state)
        fake = extract_params_from_state(self.opt['fake'], state)
        if self.noise:
            nreal = []
            nfake = []
            for i, t in enumerate(real):
                if isinstance(t, torch.Tensor):
                    nreal.append(t + torch.rand_like(t) * self.noise)
                    nfake.append(fake[i] + torch.rand_like(t) * self.noise)
                else:
                    nreal.append(t)
                    nfake.append(fake[i])
            real = nreal
            fake = nfake
        with autocast(enabled=self.env['opt']['fp16']):
            if self.opt['gan_type'] in ['gan', 'pixgan', 'pixgan_fea']:
                pred_g_fake = netD(*fake)
                loss = self.criterion(pred_g_fake, True)
            elif self.opt['gan_type'] == 'ragan':
                pred_d_real = netD(*real)
                if self.detach_real:
                    pred_d_real = pred_d_real.detach()
                pred_g_fake = netD(*fake)
                d_fake_diff = pred_g_fake - torch.mean(pred_d_real)
                self.metrics.append(("d_fake", torch.mean(pred_g_fake)))
                self.metrics.append(("d_fake_diff", torch.mean(d_fake_diff)))
                loss = (self.criterion(pred_d_real - torch.mean(pred_g_fake), False) +
                        self.criterion(d_fake_diff, True)) / 2
            else:
                raise NotImplementedError
        if self.min_loss != 0:
            self.loss_rotating_buffer[self.rb_ptr] = loss.item()
            self.rb_ptr = (self.rb_ptr + 1) % self.loss_rotating_buffer.shape[0]
            if torch.mean(self.loss_rotating_buffer) < self.min_loss:
                return 0
            self.losses_computed += 1
            self.metrics.append(("loss_counter", self.losses_computed))
        return loss


class DiscriminatorGanLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(DiscriminatorGanLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = GANLoss(opt['gan_type'], 1.0, 0.0).to(env['device'])
        self.noise = None if 'noise' not in opt.keys() else opt['noise']
        # This is a mechanism to prevent backpropagation for a GAN loss if it goes too low. This can be used to balance
        # generators and discriminators by essentially having them skip steps while their counterparts "catch up".
        self.min_loss = opt['min_loss'] if 'min_loss' in opt.keys() else 0
        self.gradient_penalty = opt['gradient_penalty'] if 'gradient_penalty' in opt.keys() else False
        if self.min_loss != 0:
            assert not self.env['dist']  # distributed training does not support 'min_loss' - it can result in backward() desync by design.
            self.loss_rotating_buffer = torch.zeros(10, requires_grad=False)
            self.rb_ptr = 0
            self.losses_computed = 0

    def forward(self, net, state):
        real = extract_params_from_state(self.opt['real'], state)
        real = [r.detach() for r in real]
        if self.gradient_penalty:
            [r.requires_grad_() for r in real]
        fake = extract_params_from_state(self.opt['fake'], state)
        new_state = {}
        fake = [f.detach() for f in fake]
        new_state = {}
        if self.noise:
            nreal = []
            nfake = []
            for i, t in enumerate(real):
                if isinstance(t, torch.Tensor):
                    nreal.append(t + torch.rand_like(t) * self.noise)
                    nfake.append(fake[i] + torch.rand_like(t) * self.noise)
                else:
                    nreal.append(t)
                    nfake.append(fake[i])
            real = nreal
            fake = nfake
        with autocast(enabled=self.env['opt']['fp16']):
            d_real = net(*real)
            d_fake = net(*fake)

        if self.opt['gan_type'] in ['gan', 'pixgan']:
            self.metrics.append(("d_fake", torch.mean(d_fake)))
            self.metrics.append(("d_real", torch.mean(d_real)))
            l_real = self.criterion(d_real, True)
            l_fake = self.criterion(d_fake, False)
            l_total = l_real + l_fake
            loss = l_total
        elif self.opt['gan_type'] == 'ragan' or self.opt['gan_type'] == 'max_spread':
            d_fake_diff = d_fake - torch.mean(d_real)
            self.metrics.append(("d_fake_diff", torch.mean(d_fake_diff)))
            loss = (self.criterion(d_real - torch.mean(d_fake), True) +
                    self.criterion(d_fake_diff, False))
        else:
            raise NotImplementedError
        if self.min_loss != 0:
            self.loss_rotating_buffer[self.rb_ptr] = loss.item()
            self.rb_ptr = (self.rb_ptr + 1) % self.loss_rotating_buffer.shape[0]
            self.metrics.append(("loss_counter", self.losses_computed))
            if torch.mean(self.loss_rotating_buffer) < self.min_loss:
                return 0
            self.losses_computed += 1

        if self.gradient_penalty:
            # Apply gradient penalty. TODO: migrate this elsewhere.
            from models.image_generation.stylegan.stylegan2_lucidrains import gradient_penalty
            assert len(real) == 1   # Grad penalty doesn't currently support multi-input discriminators.
            gp, gp_structure = gradient_penalty(real[0], d_real, return_structured_grads=True)
            self.metrics.append(("gradient_penalty", gp.clone().detach()))
            loss = loss + gp
            self.metrics.append(("gradient_penalty", gp))
            # The gp_structure is a useful visual debugging tool to see what areas of the generated image the disc is paying attention to.
            gpimg = (gp_structure / (torch.std(gp_structure, dim=(-1, -2), keepdim=True) * 2)) \
                              - torch.mean(gp_structure, dim=(-1, -2), keepdim=True) + .5
            new_state['%s_%s_gp_structure_img' % (self.opt['fake'], self.opt['real'])] = gpimg

        return loss, new_state

    # This loss is stateful because it injects a debugging result from the GP term when enabled.
    def is_stateful(self) -> bool:
        return True


# Computes a loss created by comparing the output of a generator to the output from the same generator when fed an
# input that has been altered randomly by rotation or flip.
# The "real" parameter to this loss is the actual output of the generator (from an injection point)
# The "fake" parameter is the LR input that produced the "real" parameter when fed through the generator.
class GeometricSimilarityGeneratorLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(GeometricSimilarityGeneratorLoss, self).__init__(opt, env)
        self.opt = opt
        self.generator = opt['generator']
        self.criterion = get_basic_criterion_for_name(opt['criterion'], env['device'])
        self.gen_input_for_alteration = opt['input_alteration_index'] if 'input_alteration_index' in opt.keys() else 0
        self.gen_output_to_use = opt['generator_output_index'] if 'generator_output_index' in opt.keys() else None
        self.detach_fake = opt['detach_fake'] if 'detach_fake' in opt.keys() else False

    # Returns a random alteration and its counterpart (that undoes the alteration)
    def random_alteration(self):
        return random.choice([(functools.partial(torch.flip, dims=(2,)), functools.partial(torch.flip, dims=(2,))),
                              (functools.partial(torch.flip, dims=(3,)), functools.partial(torch.flip, dims=(3,))),
                              (functools.partial(torch.rot90, k=1, dims=[2,3]), functools.partial(torch.rot90, k=3, dims=[2,3])),
                              (functools.partial(torch.rot90, k=2, dims=[2,3]), functools.partial(torch.rot90, k=2, dims=[2,3])),
                              (functools.partial(torch.rot90, k=3, dims=[2,3]), functools.partial(torch.rot90, k=1, dims=[2,3]))])

    def forward(self, net, state):
        net = self.env['generators'][self.generator]  # Get the network from an explicit parameter.
                                                    # The <net> parameter is not reliable for generator losses since often they are combined with many networks.
        fake = extract_params_from_state(self.opt['fake'], state)
        alteration, undo_fn = self.random_alteration()
        altered = []
        for i, t in enumerate(fake):
            if i == self.gen_input_for_alteration:
                altered.append(alteration(t))
            else:
                altered.append(t)

        with autocast(enabled=self.env['opt']['fp16']):
            if self.detach_fake:
                with torch.no_grad():
                    upsampled_altered = net(*altered)
            else:
                upsampled_altered = net(*altered)

        if self.gen_output_to_use is not None:
            upsampled_altered = upsampled_altered[self.gen_output_to_use]

        # Undo alteration on HR image
        upsampled_altered = undo_fn(upsampled_altered)

        if self.opt['criterion'] == 'cosine':
            return self.criterion(state[self.opt['real']], upsampled_altered, torch.ones(1, device=upsampled_altered.device))
        else:
            return self.criterion(state[self.opt['real']].float(), upsampled_altered.float())


# Computes a loss created by comparing the output of a generator to the output from the same generator when fed an
# input that has been translated in a random direction.
# The "real" parameter to this loss is the actual output of the generator on the top left image patch.
# The "fake" parameter is the output base fed into a ImagePatchInjector.
class TranslationInvarianceLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(TranslationInvarianceLoss, self).__init__(opt, env)
        self.opt = opt
        self.generator = opt['generator']
        self.criterion = get_basic_criterion_for_name(opt['criterion'], env['device'])
        self.gen_input_for_alteration = opt['input_alteration_index'] if 'input_alteration_index' in opt.keys() else 0
        self.gen_output_to_use = opt['generator_output_index'] if 'generator_output_index' in opt.keys() else None
        self.patch_size = opt['patch_size']
        self.overlap = opt['overlap']  # For maximum overlap, can be calculated as 2*patch_size-image_size
        self.detach_fake = opt['detach_fake']
        assert(self.patch_size > self.overlap)

    def forward(self, net, state):
        net = self.env['generators'][self.generator]  # Get the network from an explicit parameter.
        # The <net> parameter is not reliable for generator losses since often they are combined with many networks.

        border_sz = self.patch_size - self.overlap
        translation = random.choice([("top_right", border_sz, border_sz+self.overlap, 0, self.overlap),
                                 ("bottom_left", 0, self.overlap, border_sz, border_sz+self.overlap),
                                 ("bottom_right", 0, self.overlap, 0, self.overlap)])
        trans_name, hl, hh, wl, wh = translation
        # Change the "fake" input name that we are translating to one that specifies the random translation.
        fake = self.opt['fake'].copy()
        fake[self.gen_input_for_alteration] = "%s_%s" % (fake[self.gen_input_for_alteration], trans_name)
        input = extract_params_from_state(fake, state)

        with autocast(enabled=self.env['opt']['fp16']):
            if self.detach_fake:
                with torch.no_grad():
                    trans_output = net(*input)
            else:
                trans_output = net(*input)
        if not isinstance(trans_output, list) and not isinstance(trans_output, tuple):
            trans_output = [trans_output]

        if self.gen_output_to_use is not None:
            fake_shared_output = trans_output[self.gen_output_to_use][:, :, hl:hh, wl:wh]
        else:
            fake_shared_output = trans_output[:, :, hl:hh, wl:wh]

        # The "real" input is assumed to always come from the top left tile.
        gen_output = state[self.opt['real']]
        real_shared_output = gen_output[:, :, border_sz:border_sz+self.overlap, border_sz:border_sz+self.overlap]

        if self.opt['criterion'] == 'cosine':
            return self.criterion(fake_shared_output, real_shared_output, torch.ones(1, device=real_shared_output.device))
        else:
            return self.criterion(fake_shared_output.float(), real_shared_output.float())


# Computes a loss repeatedly feeding the generator downsampled inputs created from its outputs. The expectation is
# that the generator's outputs do not change on repeated forward passes.
# The "real" parameter to this loss is the actual output of the generator.
# The "fake" parameter is the expected inputs that should be fed into the generator. 'input_alteration_index' is changed
#   so that it feeds the recursive input.
class RecursiveInvarianceLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(RecursiveInvarianceLoss, self).__init__(opt, env)
        self.opt = opt
        self.generator = opt['generator']
        self.criterion = get_basic_criterion_for_name(opt['criterion'], env['device'])
        self.gen_input_for_alteration = opt['input_alteration_index'] if 'input_alteration_index' in opt.keys() else 0
        self.gen_output_to_use = opt['generator_output_index'] if 'generator_output_index' in opt.keys() else None
        self.recursive_depth = opt['recursive_depth']  # How many times to recursively feed the output of the generator back into itself
        self.downsample_factor = opt['downsample_factor']  # Just 1/opt['scale']. Necessary since this loss doesnt have access to opt['scale'].
        assert(self.recursive_depth > 0)

    def forward(self, net, state):
        net = self.env['generators'][self.generator]  # Get the network from an explicit parameter.
        # The <net> parameter is not reliable for generator losses since they can be combined with many networks.

        gen_output = state[self.opt['real']]
        recurrent_gen_output = gen_output

        fake = self.opt['fake'].copy()
        input = extract_params_from_state(fake, state)
        for i in range(self.recursive_depth):
            input[self.gen_input_for_alteration] = torch.nn.functional.interpolate(recurrent_gen_output, scale_factor=self.downsample_factor, mode="nearest")
            with autocast(enabled=self.env['opt']['fp16']):
                recurrent_gen_output = net(*input)[self.gen_output_to_use]

        compare_real = gen_output
        compare_fake = recurrent_gen_output
        if self.opt['criterion'] == 'cosine':
            return self.criterion(compare_real, compare_fake, torch.ones(1, device=compare_real.device))
        else:
            return self.criterion(compare_real.float(), compare_fake.float())


# Loss that pulls tensors from dim 1 of the input and repeatedly feeds them into the
# 'subtype' loss.
class RecurrentLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(RecurrentLoss, self).__init__(opt, env)
        o = opt.copy()
        o['type'] = opt['subtype']
        o['fake'] = '_fake'
        o['real'] = '_real'
        self.loss = create_loss(o, self.env)
        # Use this option to specify a differential weighting scheme for losses inside of the recurrent construct. For
        # example, if later recurrent outputs should contribute more to the loss than earlier ones. When specified,
        # must be a list of weights that exactly aligns with the recurrent list fed to forward().
        self.recurrent_weights = opt['recurrent_weights'] if 'recurrent_weights' in opt.keys() else 1

    def forward(self, net, state):
        total_loss = 0
        st = state.copy()
        real = state[self.opt['real']]
        for i in range(real.shape[1]):
            st['_real'] = real[:, i]
            st['_fake'] = state[self.opt['fake']][:, i]
            subloss = self.loss(net, st)
            if isinstance(self.recurrent_weights, list):
                subloss = subloss * self.recurrent_weights[i]
            total_loss += subloss
        return total_loss

    def extra_metrics(self):
        return self.loss.extra_metrics()

    def clear_metrics(self):
        self.loss.clear_metrics()


# Loss that pulls a tensor from dim 1 of the input and feeds it into a "sub" loss.
class ForElementLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(ForElementLoss, self).__init__(opt, env)
        o = opt.copy()
        o['type'] = opt['subtype']
        self.index = opt['index']
        o['fake'] = '_fake'
        o['real'] = '_real'
        self.loss = create_loss(o, self.env)

    def forward(self, net, state):
        st = state.copy()
        st['_real'] = state[self.opt['real']][:, self.index]
        st['_fake'] = state[self.opt['fake']][:, self.index]
        return self.loss(net, st)

    def extra_metrics(self):
        return self.loss.extra_metrics()

    def clear_metrics(self):
        self.loss.clear_metrics()
