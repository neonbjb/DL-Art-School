import copy
import os
from functools import wraps
import kornia.augmentation as augs

import torch
import torch.nn.functional as F
import torchvision
from kornia import filters
from torch import nn

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

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        unused = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)
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

        augmentations = [ \
            RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
            augs.RandomGrayscale(p=0.2),
            augs.RandomHorizontalFlip(),
            RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
            augs.RandomResizedCrop((image_size, image_size))]
        self.aug = nn.Sequential(*augmentations)
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device),
                     torch.randn(2, 3, image_size, image_size, device=device))

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

    def forward(self, image_one, image_two):
        image_one = self.aug(image_one.clone())
        image_two = self.aug(image_two.clone())

        # Keep copies on hand for visual_dbg.
        self.im1 = image_one.detach().clone()
        self.im2 = image_two.detach().clone()

        online_proj_one = self.online_encoder(image_one)
        online_proj_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one = target_encoder(image_one).detach()
            target_proj_two = target_encoder(image_two).detach()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()


@register_model
def register_byol(opt_net, opt):
    subnet = create_model(opt, opt_net['subnet'])
    return BYOL(subnet, opt_net['image_size'], opt_net['hidden_layer'],
                structural_mlp=opt_get(opt_net, ['use_structural_mlp'], False))