import math

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from models.segformer.backbone import backbone50


# torch.gather() which operates as it always fucking should have: pulling indexes from the input.
from trainer.networks import register_model


def gather_2d(input, index):
    b, c, h, w = input.shape
    nodim = input.view(b, c, h * w)
    ind_nd = index[:, 0]*w + index[:, 1]
    ind_nd = ind_nd.unsqueeze(1)
    ind_nd = ind_nd.repeat((1, c))
    ind_nd = ind_nd.unsqueeze(2)
    result = torch.gather(nodim, dim=2, index=ind_nd)
    result = result.squeeze()
    if b == 1:
        result = result.unsqueeze(0)
    return result


class DilatorModule(nn.Module):
    def __init__(self, input_channels, output_channels, max_dilation):
        super().__init__()
        self.max_dilation = max_dilation
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, dilation=1, bias=True)
        if max_dilation > 1:
            self.bn = nn.BatchNorm2d(input_channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=max_dilation, dilation=max_dilation, bias=True)
        self.dense = nn.Linear(input_channels, output_channels, bias=True)

    def forward(self, inp, loc):
        x = self.conv1(inp)
        if self.max_dilation > 1:
            x = self.bn(self.relu(x))
            x = self.conv2(x)

        # This can be made more efficient by only computing these convolutions across a subset of the image. Possibly.
        x = gather_2d(x, loc).contiguous()
        return self.dense(x)


# Grabbed from torch examples: https://github.com/pytorch/examples/tree/master/https://github.com/pytorch/examples/blob/master/word_language_model/model.py#L65:7
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


# Simple mean() layer encoded into a class so that BYOL can grab it.
class Tail(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=0)


class Segformer(nn.Module):
    def __init__(self, latent_channels=1024, layers=8):
        super().__init__()
        self.backbone = backbone50()
        backbone_channels = [256, 512, 1024, 2048]
        dilations = [[1,2,3,4],[1,2,3],[1,2],[1]]
        final_latent_channels = latent_channels
        dilators = []
        for ic, dis in zip(backbone_channels, dilations):
            layer_dilators = []
            for di in dis:
                layer_dilators.append(DilatorModule(ic, final_latent_channels, di))
            dilators.append(nn.ModuleList(layer_dilators))
        self.dilators = nn.ModuleList(dilators)

        self.token_position_encoder = PositionalEncoding(final_latent_channels, max_len=10)
        self.transformer_layers = nn.Sequential(*[nn.TransformerEncoderLayer(final_latent_channels, nhead=4) for _ in range(layers)])
        self.tail = Tail()

    def forward(self, img=None, layers=None, pos=None, return_layers=False):
        assert img is not None or layers is not None
        if img is not None:
            bs = img.shape[0]
            layers = self.backbone(img)
        else:
            bs = layers[0].shape[0]
        if return_layers:
            return layers

        # A single position can be optionally given, in which case we need to expand it to represent the entire input.
        if pos.shape == (2,):
            pos = pos.unsqueeze(0).repeat(bs, 1)

        set = []
        pos = pos // 4
        for layer_out, dilator in zip(layers, self.dilators):
            for subdilator in dilator:
                set.append(subdilator(layer_out, pos))
            pos = pos // 2

        # The torch transformer expects the set dimension to be 0.
        set = torch.stack(set, dim=0)
        set = self.token_position_encoder(set)
        set = self.transformer_layers(set)
        return self.tail(set)


@register_model
def register_segformer(opt_net, opt):
    return Segformer()


if __name__ == '__main__':
    model = Segformer().to('cuda')
    for j in tqdm(range(1000)):
        test_tensor = torch.randn(64,3,224,224).cuda()
        print(model(img=test_tensor, pos=torch.randint(0,224,(64,2)).cuda()).shape)