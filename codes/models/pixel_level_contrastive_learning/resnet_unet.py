# Resnet implementation that adds a u-net style up-conversion component to output values at a
# specified pixel density.
#
# The downsampling part of the network is compatible with the built-in torch resnet for use in
# transfer learning.
#
# Only resnet50 currently supported.

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3
from torchvision.models.utils import load_state_dict_from_url
import torchvision


from trainer.networks import register_model
from utils.util import checkpoint

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class ReverseBottleneck(nn.Module):

    def __init__(self, inplanes, planes, groups=1, passthrough=False,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.passthrough = passthrough
        if passthrough:
            self.integrate = conv1x1(inplanes*2, inplanes)
            self.bn_integrate = norm_layer(inplanes)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, groups, dilation)
        self.bn2 = norm_layer(width)
        self.residual_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv1x1(width, width),
            norm_layer(width),
        )
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv1x1(inplanes, planes),
            norm_layer(planes),
        )

    def forward(self, x, passthrough=None):
        if self.passthrough:
            x = self.bn_integrate(self.integrate(torch.cat([x, passthrough], dim=1)))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.residual_upsample(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.upsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class UResNet50(torchvision.models.resnet.ResNet):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
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
        for i in range(2):
            uplayers.append(ReverseBottleneck(inplanes, inplanes // 2, norm_layer=norm_layer, passthrough=not first))
            inplanes = inplanes // 2
            first = False
        self.uplayers = nn.ModuleList(uplayers)
        self.tail = nn.Sequential(conv1x1(1024, 512),
                                  norm_layer(512),
                                  nn.ReLU(),
                                  conv3x3(512, 512),
                                  norm_layer(512),
                                  nn.ReLU(),
                                  conv1x1(512, 128))

        del self.fc  # Not used in this implementation and just consumes a ton of GPU memory.


    def _forward_impl(self, x):
        # Should be the exact same implementation of torchvision.models.resnet.ResNet.forward_impl,
        # except using checkpoints on the body conv layers.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = checkpoint(self.layer1, x)
        x2 = checkpoint(self.layer2, x1)
        x3 = checkpoint(self.layer3, x2)
        x4 = checkpoint(self.layer4, x3)
        unused = self.avgpool(x4)  # This is performed for instance-level pixpro learning, even though it is unused.

        x = checkpoint(self.uplayers[0], x4)
        x = checkpoint(self.uplayers[1], x, x3)
        #x = checkpoint(self.uplayers[2], x, x2)
        #x = checkpoint(self.uplayers[3], x, x1)

        return checkpoint(self.tail, torch.cat([x, x2], dim=1))

    def forward(self, x):
        return self._forward_impl(x)


@register_model
def register_u_resnet50(opt_net, opt):
    model = UResNet50(Bottleneck, [3, 4, 6, 3])
    return model


if __name__ == '__main__':
    model = UResNet50(Bottleneck, [3,4,6,3])
    samp = torch.rand(1,3,224,224)
    model(samp)
    # For pixpro: attach to "tail.3"
