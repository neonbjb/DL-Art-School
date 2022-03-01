import random

import torch.nn
from kornia.augmentation import RandomResizedCrop
from torch.cuda.amp import autocast

from trainer.inject import Injector, create_injector
from trainer.losses import extract_params_from_state
from utils.util import opt_get
from utils.weight_scheduler import get_scheduler_for_opt


# Transfers the state in the input key to the output key
class DirectInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)

    def forward(self, state):
        return {self.output: state[self.input]}


# Allows multiple injectors to be used on sequential steps.
class StepInterleaveInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        for inj in opt['subinjectors'].keys():
            o = opt.copy()
            o['subinjectors'] = opt['subtype']
            o['in'] = '_in'
            o['out'] = '_out'
        self.injector = create_injector(o, self.env)
        self.aslist = opt['aslist'] if 'aslist' in opt.keys() else False

    def forward(self, state):
        injs = []
        st = state.copy()
        inputs = state[self.opt['in']]
        for i in range(inputs.shape[1]):
            st['_in'] = inputs[:, i]
            injs.append(self.injector(st)['_out'])
        if self.aslist:
            return {self.output: injs}
        else:
            return {self.output: torch.stack(injs, dim=1)}


class PadInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.multiple = opt['multiple']

    def forward(self, state):
        ldim = state[self.input].shape[-1]
        mod = self.multiple-(ldim % self.multiple)
        t = state[self.input]
        if mod != 0:
            t = torch.nn.functional.pad(t, (0, mod))
        return {self.output: t}


class SqueezeInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.dim = opt['dim']

    def forward(self, state):
        return {self.output: state[self.input].squeeze(dim=self.dim)}


# Uses a generator to synthesize an image from [in] and injects the results into [out]
# Note that results are *not* detached.
class GeneratorInjector(Injector):
    def __init__(self, opt, env):
        super(GeneratorInjector, self).__init__(opt, env)
        self.grad = opt['grad'] if 'grad' in opt.keys() else True
        self.method = opt_get(opt, ['method'], None)  # If specified, this method is called instead of __call__()
        self.args = opt_get(opt, ['args'], {})
        self.fp16_override = opt_get(opt, ['fp16'], True)

    def forward(self, state):
        gen = self.env['generators'][self.opt['generator']]

        if self.method is not None and hasattr(gen, 'module'):
            gen = gen.module  # Dereference DDP wrapper.
        method = gen if self.method is None else getattr(gen, self.method)

        with autocast(enabled=self.env['opt']['fp16'] and self.fp16_override):
            if isinstance(self.input, list):
                params = extract_params_from_state(self.input, state)
            else:
                params = [state[self.input]]
            if self.grad:
                results = method(*params, **self.args)
            else:
                was_training = gen.training
                gen.eval()
                with torch.no_grad():
                    results = method(*params, **self.args)
                if was_training:
                    gen.train()
        new_state = {}
        if isinstance(self.output, list):
            # Only dereference tuples or lists, not tensors. IF YOU REACH THIS ERROR, REMOVE THE BRACES AROUND YOUR OUTPUTS IN THE YAML CONFIG
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
        self.mode = opt['mode'] if 'mode' in opt.keys() else 'normal'

    def forward(self, state):
        # Scale can be a fixed float, or a state key (e.g. from ScheduledScalarInjector).
        if isinstance(self.opt['scale'], str):
            scale = state[self.opt['scale']]
        else:
            scale = self.opt['scale']
            if scale is None:
                scale = 1

        ref = state[self.opt['in']]
        if self.mode == 'normal':
            noise = torch.randn_like(ref) * scale
        elif self.mode == 'uniform':
            noise = torch.FloatTensor(ref.shape).uniform_(0.0, scale).to(ref.device)
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
        self.resize = opt[
            'resize'] if 'resize' in opt.keys() else None  # If specified, the output is resized to a square with this size after patch extraction.

    def forward(self, state):
        im = state[self.opt['in']]
        if self.env['training']:
            res = {self.opt['out']: im[:, :3, :self.patch_size, :self.patch_size],
                   '%s_top_left' % (self.opt['out'],): im[:, :, :self.patch_size, :self.patch_size],
                   '%s_top_right' % (self.opt['out'],): im[:, :, :self.patch_size, -self.patch_size:],
                   '%s_bottom_left' % (self.opt['out'],): im[:, :, -self.patch_size:, :self.patch_size],
                   '%s_bottom_right' % (self.opt['out'],): im[:, :, -self.patch_size:, -self.patch_size:]}
        else:
            res = {self.opt['out']: im,
                   '%s_top_left' % (self.opt['out'],): im,
                   '%s_top_right' % (self.opt['out'],): im,
                   '%s_bottom_left' % (self.opt['out'],): im,
                   '%s_bottom_right' % (self.opt['out'],): im}
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
                output.append(input[b, :, self.margin + shiftleft:-(self.margin - shiftleft),
                              self.margin + shifttop:-(self.margin - shifttop)])
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
        self.aslist = opt['aslist'] if 'aslist' in opt.keys() else False

    def forward(self, state):
        injs = []
        st = state.copy()
        inputs = state[self.opt['in']]
        for i in range(inputs.shape[1]):
            st['_in'] = inputs[:, i]
            injs.append(self.injector(st)['_out'])
        if self.aslist:
            return {self.output: injs}
        else:
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
        return {self.opt['out']: out}


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


class BatchRotateInjector(Injector):
    def __init__(self, opt, env):
        super(BatchRotateInjector, self).__init__(opt, env)

    def forward(self, state):
        img = state[self.input]
        return {self.output: torch.roll(img, 1, 0)}


# Injector used to work with image deltas used in diff-SR
class SrDiffsInjector(Injector):
    def __init__(self, opt, env):
        super(SrDiffsInjector, self).__init__(opt, env)
        self.mode = opt['mode']
        assert self.mode in ['recombine', 'produce_diff']
        self.lq = opt['lq']
        self.hq = opt['hq']
        if self.mode == 'produce_diff':
            self.diff_key = opt['diff']
            self.include_combined = opt['include_combined']

    def forward(self, state):
        resampled_lq = state[self.lq]
        hq = state[self.hq]
        if self.mode == 'produce_diff':
            diff = hq - resampled_lq
            if self.include_combined:
                res = torch.cat([resampled_lq, diff, hq], dim=1)
            else:
                res = torch.cat([resampled_lq, diff], dim=1)
            return {self.output: res,
                    self.diff_key: diff}
        elif self.mode == 'recombine':
            combined = resampled_lq + hq
            return {self.output: combined}


class MultiFrameCombiner(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.mode = opt['mode']
        self.dim = opt['dim'] if 'dim' in opt.keys() else None
        self.flow = opt['flow']
        self.in_lq_key = opt['in']
        self.in_hq_key = opt['in_hq']
        self.out_lq_key = opt['out']
        self.out_hq_key = opt['out_hq']
        from models.flownet2.networks import Resample2d
        self.resampler = Resample2d()

    def combine(self, state):
        flow = self.env['generators'][self.flow]
        lq = state[self.in_lq_key]
        hq = state[self.in_hq_key]
        b, f, c, h, w = lq.shape
        center = f // 2
        center_img = lq[:, center, :, :, :]
        imgs = [center_img]
        with torch.no_grad():
            for i in range(f):
                if i == center:
                    continue
                nimg = lq[:, i, :, :, :]
                flowfield = flow(torch.stack([center_img, nimg], dim=2).float())
                nimg = self.resampler(nimg, flowfield)
                imgs.append(nimg)
        hq_out = hq[:, center, :, :, :]
        return {self.out_lq_key: torch.cat(imgs, dim=1),
                self.out_hq_key: hq_out,
                self.out_lq_key + "_flow_sample": torch.cat(imgs, dim=0)}

    def synthesize(self, state):
        lq = state[self.in_lq_key]
        return {
            self.out_lq_key: lq.repeat(1, self.dim, 1, 1)
        }

    def forward(self, state):
        if self.mode == "synthesize":
            return self.synthesize(state)
        elif self.mode == "combine":
            return self.combine(state)
        else:
            raise NotImplementedError


# Combines data from multiple different sources and mixes them along the batch dimension. Labels are then emitted
# according to how the mixing was performed.
class MixAndLabelInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.out_labels = opt['out_labels']

    def forward(self, state):
        input_tensors = [state[i] for i in self.input]
        num_inputs = len(input_tensors)
        bs = input_tensors[0].shape[0]
        labels = torch.randint(0, num_inputs, (bs,), device=input_tensors[0].device)
        # Still don't know of a good way to do this in torch.. TODO make it better..
        res = []
        for b in range(bs):
            res.append(input_tensors[labels[b]][b, :, :, :])
        output = torch.stack(res, dim=0)
        return {self.out_labels: labels, self.output: output}


# Randomly performs a uniform resize & crop from a base image.
# Never resizes below input resolution or messes with the aspect ratio.
class RandomCropInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        dim_in = opt['dim_in']
        dim_out = opt['dim_out']
        scale = dim_out / dim_in
        self.operator = RandomResizedCrop(size=(dim_out, dim_out), scale=(scale, 1),
                                          ratio=(.99,1),  # An aspect ratio range is required, but .99,1 is effectively "none".
                                          resample='NEAREST')

    def forward(self, state):
        return {self.output: self.operator(state[self.input])}


class Stylegan2NoiseInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.mix_prob = opt_get(opt, ['mix_probability'], .9)
        self.latent_dim = opt_get(opt, ['latent_dim'], 512)

    def make_noise(self, batch, latent_dim, n_noise, device):
        return torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    def forward(self, state):
        i = state[self.input]
        if self.mix_prob > 0 and random.random() < self.mix_prob:
            return {self.output: self.make_noise(i.shape[0], self.latent_dim, 2, i.device)}
        else:
            return {self.output: self.make_noise(i.shape[0], self.latent_dim, 1, i.device)}


class NoiseInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.shape = tuple(opt['shape'])

    def forward(self, state):
        shape = (state[self.input].shape[0],) + self.shape
        return {self.output: torch.randn(shape, device=state[self.input].device)}


# Incorporates the specified dimension into the batch dimension.
class DecomposeDimensionInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.dim = opt['dim']
        self.cutoff_dim = opt_get(opt, ['cutoff_dim'], -1)
        assert self.dim != 0  # Cannot decompose the batch dimension

    def forward(self, state):
        inp = state[self.input]
        dims = list(range(len(inp.shape)))  # Looks like [0,1,2,3]
        shape = list(inp.shape)
        del dims[self.dim]
        del shape[self.dim]

        # Compute the reverse permutation and shape arguments needed to undo this operation.
        rev_shape = [inp.shape[self.dim]] + shape.copy()
        rev_permute = list(range(len(inp.shape)))[1:]  # Looks like [1,2,3]
        rev_permute = rev_permute[:self.dim] + [0] + (rev_permute[self.dim:] if self.dim < len(rev_permute) else [])

        out = inp.permute([self.dim] + dims).reshape((-1,) + tuple(shape[1:]))
        if self.cutoff_dim > -1:
            out = out[:self.cutoff_dim]

        return {self.output: out,
                f'{self.output}_reverse_shape': rev_shape,
                f'{self.output}_reverse_permute': rev_permute}


# Undoes a decompose.
class RecomposeDimensionInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.rev_shape_key = opt['reverse_shape']
        self.rev_permute_key = opt['reverse_permute']

    def forward(self, state):
        inp = state[self.input]
        rev_shape = state[self.rev_shape_key]
        rev_permute = state[self.rev_permute_key]
        out = inp.reshape(rev_shape)
        out = out.permute(rev_permute).contiguous()
        return {self.output: out}


# Performs normalization across fixed constants.
class NormalizeInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.shift = opt['shift']
        self.scale = opt['scale']

    def forward(self, state):
        inp = state[self.input]
        out = (inp - self.shift) / self.scale
        return {self.output: out}


# Performs frequency-bin normalization for spectrograms.
class FrequencyBinNormalizeInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.shift, self.scale = torch.load(opt['stats_file'])
        self.shift = self.shift.view(1,-1,1)
        self.scale = self.scale.view(1,-1,1)

    def forward(self, state):
        inp = state[self.input]
        self.shift = self.shift.to(inp.device)
        self.scale = self.scale.to(inp.device)
        out = (inp - self.shift) / self.scale
        return {self.output: out}


# Performs normalization across fixed constants.
class DenormalizeInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.shift = opt['shift']
        self.scale = opt['scale']

    def forward(self, state):
        inp = state[self.input]
        out = inp * self.scale + self.shift
        return {self.output: out}