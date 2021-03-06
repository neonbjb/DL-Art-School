import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3
from torchvision.models.utils import load_state_dict_from_url
import torchvision

from models.arch_util import ConvBnRelu
from models.pixel_level_contrastive_learning.resnet_unet import ReverseBottleneck
from trainer.networks import register_model
from utils.util import checkpoint, opt_get


class UResNet50_3(torchvision.models.resnet.ResNet):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, out_dim=128):
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation, norm_layer)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        '''
        # For reference:
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        '''
        uplayers = []
        inplanes = 2048
        first = True
        for i in range(3):
            uplayers.append(ReverseBottleneck(inplanes, inplanes // 2, norm_layer=norm_layer, passthrough=not first))
            inplanes = inplanes // 2
            first = False
        self.uplayers = nn.ModuleList(uplayers)

        # These two variables are separated out and renamed so that I can re-use parameters from a pretrained resnet_unet2.
        self.last_uplayer = ReverseBottleneck(256, 128, norm_layer=norm_layer, passthrough=True)
        self.tail3 = nn.Sequential(conv1x1(192, 128),
                                  norm_layer(128),
                                  nn.ReLU(),
                                  conv1x1(128, out_dim))

        del self.fc  # Not used in this implementation and just consumes a ton of GPU memory.


    def _forward_impl(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x0)

        x1 = checkpoint(self.layer1, x)
        x2 = checkpoint(self.layer2, x1)
        x3 = checkpoint(self.layer3, x2)
        x4 = checkpoint(self.layer4, x3)
        unused = self.avgpool(x4)  # This is performed for instance-level pixpro learning, even though it is unused.

        x = checkpoint(self.uplayers[0], x4)
        x = checkpoint(self.uplayers[1], x, x3)
        x = checkpoint(self.uplayers[2], x, x2)
        x = checkpoint(self.last_uplayer, x, x1)

        return checkpoint(self.tail3, torch.cat([x, x0], dim=1))

    def forward(self, x):
        return self._forward_impl(x)


@register_model
def register_u_resnet50_3(opt_net, opt):
    model = UResNet50_3(Bottleneck, [3, 4, 6, 3], out_dim=opt_net['odim'])
    if opt_get(opt_net, ['use_pretrained_base'], False):
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=True)
        model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == '__main__':
    model = UResNet50_3(Bottleneck, [3,4,6,3])
    samp = torch.rand(1,3,224,224)
    y = model(samp)
    print(y.shape)
    # For pixpro: attach to "tail.3"
