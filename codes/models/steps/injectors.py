import random

import torch.nn
from torch.cuda.amp import autocast

from models.archs.SPSR_arch import ImageGradientNoPadding
from models.archs.pytorch_ssim import SSIM
from utils.weight_scheduler import get_scheduler_for_opt
from models.steps.losses import extract_params_from_state

# Injectors are a way to sythesize data within a step that can then be used (and reused) by loss functions.
def create_injector(opt_inject, env):
    type = opt_inject['type']
    if 'teco_' in type:
        from models.steps.tecogan_losses import create_teco_injector
        return create_teco_injector(opt_inject, env)
    elif 'progressive_' in type:
        from models.steps.progressive_zoom import create_progressive_zoom_injector
        return create_progressive_zoom_injector(opt_inject, env)
    elif 'stereoscopic_' in type:
        from models.steps.stereoscopic import create_stereoscopic_injector
        return create_stereoscopic_injector(opt_inject, env)
    elif type == 'generator':
        return ImageGeneratorInjector(opt_inject, env)
    elif type == 'discriminator':
        return DiscriminatorInjector(opt_inject, env)
    elif type == 'scheduled_scalar':
        return ScheduledScalarInjector(opt_inject, env)
    elif type == 'img_grad':
        return ImageGradientInjector(opt_inject, env)
    elif type == 'add_noise':
        return AddNoiseInjector(opt_inject, env)
    elif type == 'greyscale':
        return GreyInjector(opt_inject, env)
    elif type == 'interpolate':
        return InterpolateInjector(opt_inject, env)
    elif type == 'image_patch':
        return ImagePatchInjector(opt_inject, env)
    elif type == 'concatenate':
        return ConcatenateInjector(opt_inject, env)
    elif type == 'margin_removal':
        return MarginRemoval(opt_inject, env)
    elif type == 'foreach':
        return ForEachInjector(opt_inject, env)
    elif type == 'constant':
        return ConstantInjector(opt_inject, env)
    elif type == 'fft':
        return ImageFftInjector(opt_inject, env)
    elif type == 'extract_indices':
        return IndicesExtractor(opt_inject, env)
    elif type == 'random_shift':
        return RandomShiftInjector(opt_inject, env)
    elif type == 'psnr':
        return PsnrInjector(opt_inject, env)
    elif type == 'batch_rotate':
        return BatchRotateInjector(opt_inject, env)
    else:
        raise NotImplementedError


class Injector(torch.nn.Module):
    def __init__(self, opt, env):
        super(Injector, self).__init__()
        self.opt = opt
        self.env = env
        if 'in' in opt.keys():
            self.input = opt['in']
        self.output = opt['out']

    # This should return a dict of new state variables.
    def forward(self, state):
        raise NotImplementedError

# Uses a generator to synthesize an image from [in] and injects the results into [out]
# Note that results are *not* detached.
class ImageGeneratorInjector(Injector):
    def __init__(self, opt, env):
        super(ImageGeneratorInjector, self).__init__(opt, env)

    def forward(self, state):
        gen = self.env['generators'][self.opt['generator']]
        with autocast(enabled=self.env['opt']['fp16']):
            if isinstance(self.input, list):
                params = extract_params_from_state(self.input, state)
                results = gen(*params)
            else:
                results = gen(state[self.input])
        new_state = {}
        if isinstance(self.output, list):
            # Only dereference tuples or lists, not tensors.
            assert isinstance(results, list) or isinstance(results, tuple)
            for i, k in enumerate(self.output):
                new_state[k] = results[i]
        else:
            new_state[self.output] = results

        return new_state


# Injects a result from a discriminator network into the state.
class DiscriminatorInjector(Injector):
    def __init__(self, opt, env):
        super(DiscriminatorInjector, self).__init__(opt, env)

    def forward(self, state):
        with autocast(enabled=self.env['opt']['fp16']):
            d = self.env['discriminators'][self.opt['discriminator']]
            if isinstance(self.input, list):
                params = [state[i] for i in self.input]
                results = d(*params)
            else:
                results = d(state[self.input])
        new_state = {}
        if isinstance(self.output, list):
            # Only dereference tuples or lists, not tensors.
            assert isinstance(results, list) or isinstance(results, tuple)
            for i, k in enumerate(self.output):
                new_state[k] = results[i]
        else:
            new_state[self.output] = results

        return new_state


# Creates an image gradient from [in] and injects it into [out]
class ImageGradientInjector(Injector):
    def __init__(self, opt, env):
        super(ImageGradientInjector, self).__init__(opt, env)
        self.img_grad_fn = ImageGradientNoPadding().to(env['device'])

    def forward(self, state):
        return {self.opt['out']: self.img_grad_fn(state[self.opt['in']])}


# Injects a scalar that is modulated with a specified schedule. Useful for increasing or decreasing the influence
# of something over time.
class ScheduledScalarInjector(Injector):
    def __init__(self, opt, env):
        super(ScheduledScalarInjector, self).__init__(opt, env)
        self.scheduler = get_scheduler_for_opt(opt['scheduler'])

    def forward(self, state):
        return {self.opt['out']: self.scheduler.get_weight_for_step(self.env['step'])}


# Adds gaussian noise to [in], scales it to [0,[scale]] and injects into [out]
class AddNoiseInjector(Injector):
    def __init__(self, opt, env):
        super(AddNoiseInjector, self).__init__(opt, env)

    def forward(self, state):
        # Scale can be a fixed float, or a state key (e.g. from ScheduledScalarInjector).
        if isinstance(self.opt['scale'], str):
            scale = state[self.opt['scale']]
        else:
            scale = self.opt['scale']

        noise = torch.randn_like(state[self.opt['in']], device=self.env['device']) * scale
        return {self.opt['out']: state[self.opt['in']] + noise}


# Averages the channel dimension (1) of [in] and saves to [out]. Dimensions are
# kept the same, the average is simply repeated.
class GreyInjector(Injector):
    def __init__(self, opt, env):
        super(GreyInjector, self).__init__(opt, env)

    def forward(self, state):
        mean = torch.mean(state[self.opt['in']], dim=1, keepdim=True)
        mean = mean.repeat(1, 3, 1, 1)
        return {self.opt['out']: mean}


class InterpolateInjector(Injector):
    def __init__(self, opt, env):
        super(InterpolateInjector, self).__init__(opt, env)
        if 'scale_factor' in opt.keys():
            self.scale_factor = opt['scale_factor']
            self.size = None
        else:
            self.scale_factor = None
            self.size = (opt['size'], opt['size'])

    def forward(self, state):
        scaled = torch.nn.functional.interpolate(state[self.opt['in']], scale_factor=self.opt['scale_factor'],
                                                 size=self.opt['size'], mode=self.opt['mode'])
        return {self.opt['out']: scaled}


# Extracts four patches from the input image, each a square of 'patch_size'. The input images are taken from each
# of the four corners of the image. The intent of this loss is that each patch shares some part of the input, which
# can then be used in the translation invariance loss.
#
# This injector is unique in that it does not only produce the specified output label into state. Instead it produces five
# outputs for the specified label, one for each corner of the input as well as the specified output, which is the top left
# corner. See the code below to find out how this works.
#
# Another note: this injector operates differently in eval mode (e.g. when env['training']=False) - in this case, it
# simply sets all the output state variables to the input. This is so that you can feed the output of this injector
# directly into your generator in training without affecting test performance.
class ImagePatchInjector(Injector):
    def __init__(self, opt, env):
        super(ImagePatchInjector, self).__init__(opt, env)
        self.patch_size = opt['patch_size']
        self.resize = opt['resize'] if 'resize' in opt.keys() else None  # If specified, the output is resized to a square with this size after patch extraction.

    def forward(self, state):
        im = state[self.opt['in']]
        if self.env['training']:
            res = { self.opt['out']: im[:, :3, :self.patch_size, :self.patch_size],
                     '%s_top_left' % (self.opt['out'],): im[:, :, :self.patch_size, :self.patch_size],
                     '%s_top_right' % (self.opt['out'],): im[:, :, :self.patch_size, -self.patch_size:],
                     '%s_bottom_left' % (self.opt['out'],): im[:, :, -self.patch_size:, :self.patch_size],
                     '%s_bottom_right' % (self.opt['out'],): im[:, :, -self.patch_size:, -self.patch_size:] }
        else:
            res = { self.opt['out']: im,
                     '%s_top_left' % (self.opt['out'],): im,
                     '%s_top_right' % (self.opt['out'],): im,
                     '%s_bottom_left' % (self.opt['out'],): im,
                     '%s_bottom_right' % (self.opt['out'],): im }
        if self.resize is not None:
            res2 = {}
            for k, v in res.items():
                res2[k] = torch.nn.functional.interpolate(v, size=(self.resize, self.resize), mode="nearest")
            res = res2
        return res


# Concatenates a list of tensors on the specified dimension.
class ConcatenateInjector(Injector):
    def __init__(self, opt, env):
        super(ConcatenateInjector, self).__init__(opt, env)
        self.dim = opt['dim']

    def forward(self, state):
        input = [state[i] for i in self.input]
        return {self.opt['out']: torch.cat(input, dim=self.dim)}


# Removes margins from an image.
class MarginRemoval(Injector):
    def __init__(self, opt, env):
        super(MarginRemoval, self).__init__(opt, env)
        self.margin = opt['margin']
        self.random_shift_max = opt['random_shift_max'] if 'random_shift_max' in opt.keys() else 0

    def forward(self, state):
        input = state[self.input]
        if self.random_shift_max > 0:
            output = []
            # This is a really shitty way of doing this. If it works at all, I should reconsider using Resample2D, for example.
            for b in range(input.shape[0]):
                shiftleft = random.randint(-self.random_shift_max, self.random_shift_max)
                shifttop = random.randint(-self.random_shift_max, self.random_shift_max)
                output.append(input[b, :, self.margin+shiftleft:-(self.margin-shiftleft),
                                 self.margin+shifttop:-(self.margin-shifttop)])
            output = torch.stack(output, dim=0)
        else:
            output = input[:, :, self.margin:-self.margin,
                                 self.margin:-self.margin]

        return {self.opt['out']: output}


# Produces an injection which is composed of applying a single injector multiple times across a single dimension.
class ForEachInjector(Injector):
    def __init__(self, opt, env):
        super(ForEachInjector, self).__init__(opt, env)
        o = opt.copy()
        o['type'] = opt['subtype']
        o['in'] = '_in'
        o['out'] = '_out'
        self.injector = create_injector(o, self.env)

    def forward(self, state):
        injs = []
        st = state.copy()
        inputs = state[self.opt['in']]
        for i in range(inputs.shape[1]):
            st['_in'] = inputs[:, i]
            injs.append(self.injector(st)['_out'])
        return {self.output: torch.stack(injs, dim=1)}

    
class ConstantInjector(Injector):
    def __init__(self, opt, env):
        super(ConstantInjector, self).__init__(opt, env)
        self.constant_type = opt['constant_type']
        self.like = opt['like']  # This injector uses this tensor to determine what batch size and device to use.

    def forward(self, state):
        like = state[self.like]
        if self.constant_type == 'zeroes':
            out = torch.zeros_like(like)
        else:
            raise NotImplementedError
        return { self.opt['out']: out }


class ImageFftInjector(Injector):
    def __init__(self, opt, env):
        super(ImageFftInjector, self).__init__(opt, env)
        self.is_forward = opt['forward']  # Whether to compute a forward FFT or backward.
        self.eps = 1e-100

    def forward(self, state):
        if self.forward:
            fftim = torch.rfft(state[self.input], signal_ndim=2, normalized=True)
            b, f, h, w, c = fftim.shape
            fftim = fftim.permute(0,1,4,2,3).reshape(b,-1,h,w)
            # Normalize across spatial dimension
            mean = torch.mean(fftim, dim=(0,1))
            fftim = fftim - mean
            std = torch.std(fftim, dim=(0,1))
            fftim = (fftim + self.eps) / std
            return {self.output: fftim,
                    '%s_std' % (self.output,): std,
                    '%s_mean' % (self.output,): mean}
        else:
            b, f, h, w = state[self.input].shape
            # First, de-normalize the FFT.
            mean = state['%s_mean' % (self.input,)]
            std = state['%s_std' % (self.input,)]
            fftim = state[self.input] * std + mean - self.eps
            # Second, recover the FFT dimensions from the given filters.
            fftim = fftim.reshape(b, f // 2, 2, h, w).permute(0,1,3,4,2)
            im = torch.irfft(fftim, signal_ndim=2, normalized=True)
            return {self.output: im}


class IndicesExtractor(Injector):
    def __init__(self, opt, env):
        super(IndicesExtractor, self).__init__(opt, env)
        self.dim = opt['dim']
        assert self.dim == 1  # Honestly not sure how to support an abstract dim here, so just add yours when needed.

    def forward(self, state):
        results = {}
        for i, o in enumerate(self.output):
            if self.dim == 1:
                results[o] = state[self.input][:, i]
        return results


class RandomShiftInjector(Injector):
    def __init__(self, opt, env):
        super(RandomShiftInjector, self).__init__(opt, env)

    def forward(self, state):
        img = state[self.input]
        return {self.output: img}


class PsnrInjector(Injector):
    def __init__(self, opt, env):
        super(PsnrInjector, self).__init__(opt, env)
        self.ssim = SSIM(size_average=False, raw=True)
        self.scale = opt['output_scale_divisor']
        self.exp = opt['exponent'] if 'exponent' in opt.keys() else 1

    def forward(self, state):
        img1, img2 = state[self.input[0]], state[self.input[1]]
        ssim = self.ssim(img1, img2)
        areal_se = torch.nn.functional.interpolate(ssim, scale_factor=1/self.scale,
                                                   mode="area")
        return {self.output: areal_se}


class BatchRotateInjector(Injector):
    def __init__(self, opt, env):
        super(BatchRotateInjector, self).__init__(opt, env)

    def forward(self, state):
        img = state[self.input]
        return {self.output: torch.roll(img, 1, 0)}

