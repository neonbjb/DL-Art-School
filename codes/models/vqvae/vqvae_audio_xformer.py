import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

import torch.distributed as distributed

from models.vqvae.vqvae import Quantize
from trainer.networks import register_model
from utils.util import checkpoint, opt_get


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)

        self.activation = nn.ReLU()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + src2
        src = self.norm1(src.permute(0,2,1)).permute(0,2,1)
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        src = self.norm2(src.permute(0,2,1)).permute(0,2,1)
        return src


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, output_breadth, num_layers=8, compression_factor=8):
        super().__init__()

        self.compression_factor = compression_factor
        self.pre_conv_stack = nn.Sequential(nn.Conv1d(in_channel, channel//4, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv1d(channel//4, channel//2, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(),
                                            nn.Conv1d(channel//2, channel//2, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv1d(channel//2, channel, kernel_size=3, stride=2, padding=1))
        self.norm1 = nn.BatchNorm1d(channel)
        self.positional_embeddings = PositionalEncoding(channel, max_len=output_breadth//4)
        self.encode = nn.TransformerEncoder(TransformerEncoderLayer(d_model=channel, nhead=4, dim_feedforward=channel*2), num_layers=num_layers)

    def forward(self, input):
        x = self.norm1(self.pre_conv_stack(input)).permute(0,2,1)
        x = self.positional_embeddings(x)
        x = self.encode(x)
        return x[:,:input.shape[2]//self.compression_factor,:]


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, output_breadth, num_layers=6
    ):
        super().__init__()

        self.initial_conv = nn.Conv1d(in_channel, channel, kernel_size=1)
        self.expand = output_breadth
        self.positional_embeddings = PositionalEncoding(channel, max_len=output_breadth)
        self.encode = nn.TransformerEncoder(TransformerEncoderLayer(d_model=channel, nhead=4, dim_feedforward=channel*2), num_layers=num_layers)
        self.final_conv_stack = nn.Sequential(nn.Conv1d(channel, channel, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              nn.Conv1d(channel, out_channel, kernel_size=3, padding=1))

    def forward(self, input):
        x = self.initial_conv(input.permute(0,2,1)).permute(0,2,1)
        x = nn.functional.pad(x, (0,0,0, self.expand-input.shape[1]))
        x = self.positional_embeddings(x)
        x = self.encode(x).permute(0,2,1)
        return self.final_conv_stack(x)


class VQVAE(nn.Module):
    def __init__(
        self,
        data_channels=1,
        channel=256,
        codebook_dim=256,
        codebook_size=512,
        breadth=80,
    ):
        super().__init__()

        self.enc = Encoder(data_channels, channel, breadth)
        self.quantize_dense = nn.Linear(channel, codebook_dim)
        self.quantize = Quantize(codebook_dim, codebook_size)
        self.dec = Decoder(codebook_dim, data_channels, channel, breadth)

    def forward(self, input):
        input = input.unsqueeze(1)
        quant, diff, _ = self.encode(input)
        dec = checkpoint(self.dec, quant)
        dec = dec.squeeze(1)
        return dec, diff

    def encode(self, input):
        enc = checkpoint(self.enc, input)
        quant = self.quantize_dense(enc)
        quant, diff, id = self.quantize(quant)
        diff = diff.unsqueeze(0)
        return quant, diff, id


@register_model
def register_vqvae_xform_audio(opt_net, opt):
    kw = opt_get(opt_net, ['kwargs'], {})
    vq = VQVAE(**kw)
    return vq


if __name__ == '__main__':
    model = VQVAE()
    res=model(torch.randn(4,80))
    print(res[0].shape)