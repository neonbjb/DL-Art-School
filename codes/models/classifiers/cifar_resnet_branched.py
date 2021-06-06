"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

from models.switched_conv.switched_conv_hard_routing import HardRoutingGate
from trainer.networks import register_model


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNetTail(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64
        self.conv4_x = self._make_layer(block, 128, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 256, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv4_x(x)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, num_tails=8, dropout_rate=.2):
        super().__init__()
        self.in_channels = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.conv2_x = self._make_layer(block, 32, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 64, num_block[1], 2)
        self.tails = nn.ModuleList([ResNetTail(block, num_block, 256) for _ in range(num_tails)])
        self.selector = ResNetTail(block, num_block, num_tails)
        self.selector_gate = nn.Linear(256, 1)
        self.gate = HardRoutingGate(num_tails, hard_en=True)
        self.dropout_rate = dropout_rate
        self.final_linear = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, coarse_label):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)

        keys = []
        for t in self.tails:
            keys.append(t(output))
        keys = torch.stack(keys, dim=1)

        query = self.selector(output).unsqueeze(2)
        selector = self.selector_gate(query * keys).squeeze(-1)
        if self.training and self.dropout_rate > 0:
            bs, br = selector.shape
            drop = torch.rand((bs, br), device=x.device) > self.dropout_rate
            # Ensure that there is always at least one switch left un-dropped out
            fix_blank = (drop.sum(dim=1, keepdim=True) == 0).repeat(1, br)
            drop = drop.logical_or(fix_blank)
            selector = drop * selector
        selector = self.gate(selector)
        values = self.final_linear(selector.unsqueeze(-1) * keys)

        return values.sum(dim=1)

        #bs = output.shape[0]
        #return (tailouts[coarse_label] * torch.eye(n=bs, device=x.device).view(bs,bs,1)).sum(dim=1)

@register_model
def register_cifar_resnet18_branched(opt_net, opt):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])


if __name__ == '__main__':
    model = ResNet(BasicBlock, [2,2,2,2])
    v = model(torch.randn(2,3,32,32), torch.LongTensor([4,7]))
    print(v.shape)
    l = nn.MSELoss()(v, torch.randn_like(v))
    l.backward()

