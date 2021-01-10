# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
from torch import nn
from torch.nn import functional as F

import torch.distributed as distributed

from models.vqvae.scaled_weight_conv import ScaledWeightConv, ScaledWeightConvTranspose
from trainer.networks import register_model
from utils.util import checkpoint, opt_get


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            if distributed.is_initialized() and distributed.get_world_size() > 1:
                distributed.all_reduce(embed_onehot_sum)
                distributed.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, breadth):
        super().__init__()

        self.conv = nn.ModuleList([
            nn.ReLU(inplace=True),
            ScaledWeightConv(in_channel, channel, 3, padding=1, breadth=breadth),
            nn.ReLU(inplace=True),
            ScaledWeightConv(channel, in_channel, 1, breadth=breadth),
        ])

    def forward(self, input, masks):
        out = input
        for m in self.conv:
            if isinstance(m, ScaledWeightConv):
                out = m(out, masks)
            else:
                out = m(out)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride, breadth):
        super().__init__()

        if stride == 4:
            blocks = [
                ScaledWeightConv(in_channel, channel // 2, 4, stride=2, padding=1, breadth=breadth),
                nn.ReLU(inplace=True),
                ScaledWeightConv(channel // 2, channel, 4, stride=2, padding=1, breadth=breadth),
                nn.ReLU(inplace=True),
                ScaledWeightConv(channel, channel, 3, padding=1, breadth=breadth),
            ]

        elif stride == 2:
            blocks = [
                ScaledWeightConv(in_channel, channel // 2, 4, stride=2, padding=1, breadth=breadth),
                nn.ReLU(inplace=True),
                ScaledWeightConv(channel // 2, channel, 3, padding=1, breadth=breadth),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel, breadth=breadth))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        for block in self.blocks:
            if isinstance(block, ScaledWeightConv) or isinstance(block, ResBlock):
                input = block(input, self.masks)
            else:
                input = block(input)
        return input


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, breadth
    ):
        super().__init__()

        blocks = [ScaledWeightConv(in_channel, channel, 3, padding=1, breadth=breadth)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel, breadth=breadth))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    ScaledWeightConvTranspose(channel, channel // 2, 4, stride=2, padding=1, breadth=breadth),
                    nn.ReLU(inplace=True),
                    ScaledWeightConvTranspose(
                        channel // 2, out_channel, 4, stride=2, padding=1, breadth=breadth
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                ScaledWeightConvTranspose(channel, out_channel, 4, stride=2, padding=1, breadth=breadth)
            )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        for block in self.blocks:
            if isinstance(block, ScaledWeightConvTranspose) or isinstance(block, ResBlock) \
                    or isinstance(block, ScaledWeightConv):
                input = block(input, self.masks)
            else:
                input = block(input)
        return input


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        codebook_dim=64,
        codebook_size=512,
        breadth=8,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4, breadth=breadth)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2, breadth=breadth)
        self.quantize_conv_t = ScaledWeightConv(channel, codebook_dim, 1, breadth=breadth)
        self.quantize_t = Quantize(codebook_dim, codebook_size)
        self.dec_t = Decoder(
            codebook_dim, codebook_dim, channel, n_res_block, n_res_channel, stride=2, breadth=breadth
        )
        self.quantize_conv_b = ScaledWeightConv(codebook_dim + channel, codebook_dim, 1, breadth=breadth)
        self.quantize_b = Quantize(codebook_dim, codebook_size)
        self.upsample_t = ScaledWeightConvTranspose(
            codebook_dim, codebook_dim, 4, stride=2, padding=1, breadth=breadth
        )
        self.dec = Decoder(
            codebook_dim + codebook_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
            breadth=breadth
        )

    def forward(self, input, masks):
        # This awkward injection point is necessary to enable checkpointing to work.
        for m in [self.enc_b, self.enc_t, self.dec_t, self.dec]:
            m.masks = masks

        quant_t, quant_b, diff, _, _ = self.encode(input, masks)
        dec = self.decode(quant_t, quant_b, masks)

        return dec, diff

    def encode(self, input, masks):
        enc_b = checkpoint(self.enc_b, input)
        enc_t = checkpoint(self.enc_t, enc_b)

        quant_t = self.quantize_conv_t(enc_t, masks).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = checkpoint(self.dec_t, quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b, masks).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b, masks):
        upsample_t = self.upsample_t(quant_t, masks)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = checkpoint(self.dec, quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b, masks)

        return dec


@register_model
def register_weighted_vqvae(opt_net, opt):
    kw = opt_get(opt_net, ['kwargs'], {})
    return VQVAE(**kw)
