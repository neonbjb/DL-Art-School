import copy
import os
import random
from functools import wraps
import kornia.augmentation as augs

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from kornia import filters, apply_hflip
from torch import nn
from torchvision.transforms import ToTensor

from data.byol_attachment import RandomApply
from trainer.networks import register_model, create_model
from utils.util import checkpoint, opt_get


def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# Specialized augmentor class that applies a set of image transformations on points as well, allowing one to track
# where a point in the src image is located in the dest image. Restricts transformation such that this is possible.
class PointwiseAugmentor(nn.Module):
    def __init__(self, img_size=224):
        super().__init__()
        self.jitter = RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8)
        self.gray = augs.RandomGrayscale(p=0.2)
        self.blur = RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1)
        self.rrc = augs.RandomResizedCrop((img_size, img_size), same_on_batch=True)

    # Given a point in the source image, returns the same point in the source image, given the kornia RRC params.
    def rrc_on_point(self, src_point, params):
        dh, dw = params['dst'][:,2,1]-params['dst'][:,0,1], params['dst'][:,2,0] - params['dst'][:,0,0]
        sh, sw = params['src'][:,2,1]-params['src'][:,0,1], params['src'][:,2,0] - params['src'][:,0,0]
        scale_h, scale_w = sh.float() / dh.float(), sw.float() / dw.float()
        t, l = src_point[0] - params['src'][0,0,1], src_point[1] - params['src'][0,0,0]
        t = (t.float() / scale_h[0]).long()
        l = (l.float() / scale_w[0]).long()
        return torch.stack([t,l])

    def flip_on_point(self, pt, input):
        t, l = pt[0], pt[1]
        center = input.shape[-1] // 2
        return t, 2 * center - l

    def forward(self, x, point):
        d = self.jitter(x)
        d = self.gray(d)
        will_flip = random.random() > .5
        if will_flip:
            d = apply_hflip(d)
            point = self.flip_on_point(point, x)
        d = self.blur(d)

        invalid = True
        while invalid:
            params = self.rrc.generate_parameters(d.shape)
            potential = self.rrc_on_point(point, params)
            # '10' is an arbitrary number: we want to provide some margin. Making predictions at the very edge of an image is not very useful.
            if potential[0] <= 10 or potential[1] <= 10 or potential[0] > x.shape[-2]-10 or potential[1] > x.shape[-1]-10:
                continue
            d = self.rrc(d, params=params)
            point = potential
            invalid = False

        return d, point


# loss fn
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# MLP class for projector and predictor
class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        x = flatten(x)
        return self.net(x)


# A wrapper class for training against networks that do not collapse into a small-dimensioned latent.
class StructuralMLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        b, c, h, w = dim
        flattened_dim = c * h // 4 * w // 4
        self.net = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2, use_structural_mlp=False):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.structural_mlp = use_structural_mlp

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = output

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        if self.structural_mlp:
            projector = StructuralMLP(hidden.shape, self.projection_size, self.projection_hidden_size)
        else:
            _, dim = hidden.flatten(1,-1).shape
            projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x, pt):
        if self.layer == -1:
            return self.net(x, pt)

        if not self.hook_registered:
            self._register_hook()

        unused = self.net(x, pt)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, pt):
        representation = self.get_representation(x, pt)
        projector = self._get_projector(representation)
        projection = checkpoint(projector, representation)
        return projection


class BYOL(nn.Module):
    def __init__(
            self,
            net,
            image_size,
            hidden_layer=-2,
            projection_size=256,
            projection_hidden_size=4096,
            moving_average_decay=0.99,
            use_momentum=True,
            structural_mlp=False,
    ):
        super().__init__()

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer,
                                         use_structural_mlp=structural_mlp)

        self.aug = PointwiseAugmentor(image_size)
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        for p in target_encoder.parameters():
            p.DO_NOT_TRAIN = True
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_for_step(self, step, __):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def get_debug_values(self, step, __):
        # In the BYOL paper, this is made to increase over time. Not yet implemented, but still logging the value.
        return {'target_ema_beta': self.target_ema_updater.beta}

    def visual_dbg(self, step, path):
        torchvision.utils.save_image(self.im1.cpu().float(), os.path.join(path, "%i_image1.png" % (step,)))
        torchvision.utils.save_image(self.im2.cpu().float(), os.path.join(path, "%i_image2.png" % (step,)))

    def forward(self, image):
        _, _, h, w = image.shape
        point = torch.randint(h//8, 7*h//8, (2,)).long().to(image.device)

        image_one, pt_one = self.aug(image, point)
        image_two, pt_two = self.aug(image, point)

        # Keep copies on hand for visual_dbg.
        self.im1 = image_one.detach().clone()
        self.im1[:,:,pt_one[0]-3:pt_one[0]+3,pt_one[1]-3:pt_one[1]+3] = 1
        self.im2 = image_two.detach().clone()
        self.im2[:,:,pt_two[0]-3:pt_two[0]+3,pt_two[1]-3:pt_two[1]+3] = 1

        online_proj_one = self.online_encoder(image_one, pt_one)
        online_proj_two = self.online_encoder(image_two, pt_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one = target_encoder(image_one, pt_one).detach()
            target_proj_two = target_encoder(image_two, pt_two).detach()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()


if __name__ == '__main__':
    pa = PointwiseAugmentor(256)
    for j in range(100):
        t = ToTensor()(Image.open('E:\\4k6k\\datasets\\ns_images\\imagesets\\000001_152761.jpg')).unsqueeze(0).repeat(8,1,1,1)
        p = torch.randint(50,180,(2,))
        augmented, dp = pa(t, p)
        t, p = pa(t, p)
        t[:,:,p[0]-3:p[0]+3,p[1]-3:p[1]+3] = 0
        torchvision.utils.save_image(t, f"{j}_src.png")
        augmented[:,:,dp[0]-3:dp[0]+3,dp[1]-3:dp[1]+3] = 0
        torchvision.utils.save_image(augmented, f"{j}_dst.png")


@register_model
def register_pixel_local_byol(opt_net, opt):
    subnet = create_model(opt, opt_net['subnet'])
    return BYOL(subnet, opt_net['image_size'], opt_net['hidden_layer'])