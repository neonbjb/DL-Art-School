import math

import torch
import torch.nn as nn
from tqdm import tqdm

from models.segformer.backbone import backbone50


class DilatorModule(nn.Module):
    def __init__(self, input_channels, output_channels, max_dilation):
        super().__init__()
        self.max_dilation = max_dilation
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, dilation=1, bias=True)
        if max_dilation > 1:
            self.bn = nn.BatchNorm2d(input_channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, dilation=max_dilation, bias=True)
        self.dense = nn.Linear(input_channels, output_channels, bias=True)

    def forward(self, inp, loc):
        x = self.conv1(inp)
        if self.max_dilation > 1:
            x = self.bn(self.relu(x))
            x = self.conv2(x)

        # This can be made (possibly substantially) more efficient by only computing these convolutions across a subset of the image. Possibly.
        i, j = loc
        x = x[:,:,i,j]
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


class Segformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone50()
        backbone_channels = [256, 512, 1024, 2048]
        dilations = [[1,2,3,4],[1,2,3],[1,2],[1]]
        final_latent_channels = 2048
        dilators = []
        for ic, dis in zip(backbone_channels, dilations):
            layer_dilators = []
            for di in dis:
                layer_dilators.append(DilatorModule(ic, final_latent_channels, di))
            dilators.append(nn.ModuleList(layer_dilators))
        self.dilators = nn.ModuleList(dilators)

        self.token_position_encoder = PositionalEncoding(final_latent_channels, max_len=10)
        self.transformer_layers = nn.Sequential(*[nn.TransformerEncoderLayer(final_latent_channels, nhead=4) for _ in range(16)])

    def forward(self, x, pos):
        layers = self.backbone(x)
        set = []
        i, j = pos[0] // 4, pos[1] // 4
        for layer_out, dilator in zip(layers, self.dilators):
            for subdilator in dilator:
                set.append(subdilator(layer_out, (i, j)))
            i, j = i // 2, j // 2

        # The torch transformer expects the set dimension to be 0.
        set = torch.stack(set, dim=0)
        set = self.token_position_encoder(set)
        set = self.transformer_layers(set)
        return set


if __name__ == '__main__':
    model = Segformer().to('cuda')
    for j in tqdm(range(1000)):
        test_tensor = torch.randn(64,3,224,224).cuda()
        model(test_tensor, (43, 73))