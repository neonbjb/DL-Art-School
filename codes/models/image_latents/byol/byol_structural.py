import copy

import torch
import torch.nn.functional as F
from torch import nn

from data.images.byol_attachment import reconstructed_shared_regions
from models.image_latents.byol.byol_model_wrapper import singleton, EMA, get_module_device, set_requires_grad, \
    update_moving_average
from trainer.networks import create_model, register_model
from utils.util import checkpoint

# loss function
def structural_loss_fn(x, y):
    # Combine the structural dimensions into the batch dimension, then compute the "normal" BYOL loss.
    x = x.permute(0,2,3,1).flatten(0,2)
    y = y.permute(0,2,3,1).flatten(0,2)
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class StructuralTail(nn.Module):
    def __init__(self, channels, projection_size, hidden_size=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_size, kernel_size=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, projection_size, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

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
        projector = StructuralTail(hidden.shape[1], self.projection_size, self.projection_hidden_size)
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


class StructuralBYOL(nn.Module):
    def __init__(
            self,
            net,
            image_size,
            hidden_layer=-2,
            projection_size=256,
            projection_hidden_size=512,
            moving_average_decay=0.99,
            use_momentum=True,
            pretrained_state_dict=None,
            freeze_until=0
    ):
        super().__init__()

        if pretrained_state_dict:
            net.load_state_dict(torch.load(pretrained_state_dict), strict=True)
        self.freeze_until = freeze_until
        self.frozen = False
        if self.freeze_until > 0:
            for p in net.parameters():
                p.DO_NOT_TRAIN = True
            self.frozen = True
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = StructuralTail(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device),
                     torch.randn(2, 3, image_size, image_size, device=device), None)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_for_step(self, step, __):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
        if self.frozen and self.freeze_until < step:
            print("Unfreezing model weights. Let the latent training commence..")
            for p in self.online_encoder.net.parameters():
                del p.DO_NOT_TRAIN
            self.frozen = False

    def forward(self, image_one, image_two, similar_region_params):
        online_proj_one = self.online_encoder(image_one)
        online_proj_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one = target_encoder(image_one).detach()
            target_proj_two = target_encoder(image_two).detach()

        # In the structural BYOL, only the regions of the source image that are shared between the two augments are
        # compared. These regions can be extracted from the latents using `reconstruct_shared_regions`.
        if similar_region_params is not None:
            online_pred_one, target_proj_two = reconstructed_shared_regions(online_pred_one, target_proj_two, similar_region_params)
        loss_one = structural_loss_fn(online_pred_one, target_proj_two.detach())
        if similar_region_params is not None:
            online_pred_two, target_proj_one = reconstructed_shared_regions(online_pred_two, target_proj_one, similar_region_params)
        loss_two = structural_loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()

    def get_projection(self, image):
        enc = self.online_encoder(image)
        proj = self.online_predictor(enc)
        return enc, proj

@register_model
def register_structural_byol(opt_net, opt):
    subnet = create_model(opt, opt_net['subnet'])
    return StructuralBYOL(subnet, opt_net['image_size'], opt_net['hidden_layer'],
                          pretrained_state_dict=opt_get(opt_net, ["pretrained_path"]),
                          freeze_until=opt_get(opt_net, ['freeze_until'], 0))
