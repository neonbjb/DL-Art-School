import os

import torch
import torchvision
from torch import nn
from torch.nn import functional as F

import torch.distributed as distributed

from models.switched_conv import SwitchedConv, convert_conv_net_state_dict_to_switched_conv
from trainer.networks import register_model
from utils.util import checkpoint, opt_get


# Upsamples and blurs (similar to StyleGAN). Replaces ConvTranspose2D from the original paper.
class UpsampleConv(nn.Module):
    def __init__(self, in_filters, out_filters, breadth, kernel_size, padding):
        super().__init__()
        self.conv = SwitchedConv(in_filters, out_filters, kernel_size, breadth, padding=padding, include_coupler=True, coupler_mode='lambda', coupler_dim_in=in_filters)

    def forward(self, x):
        up = torch.nn.functional.interpolate(x, scale_factor=2)
        return self.conv(up)


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

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            SwitchedConv(in_channel, channel, 3, breadth, padding=1, include_coupler=True, coupler_mode='lambda', coupler_dim_in=in_channel),
            nn.ReLU(inplace=True),
            SwitchedConv(channel, in_channel, 1, breadth, include_coupler=True, coupler_mode='lambda', coupler_dim_in=channel),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride, breadth):
        super().__init__()

        if stride == 4:
            blocks = [
                SwitchedConv(in_channel, channel // 2, 5, breadth, stride=2, padding=2, include_coupler=True, coupler_mode='lambda', coupler_dim_in=in_channel),
                nn.ReLU(inplace=True),
                SwitchedConv(channel // 2, channel, 5, breadth, stride=2, padding=2, include_coupler=True, coupler_mode='lambda', coupler_dim_in=channel // 2),
                nn.ReLU(inplace=True),
                SwitchedConv(channel, channel, 3, breadth, padding=1, include_coupler=True, coupler_mode='lambda', coupler_dim_in=channel),
            ]

        elif stride == 2:
            blocks = [
                SwitchedConv(in_channel, channel // 2, 5, breadth, stride=2, padding=2, include_coupler=True, coupler_mode='lambda', coupler_dim_in=in_channel),
                nn.ReLU(inplace=True),
                SwitchedConv(channel // 2, channel, 3, breadth, padding=1, include_coupler=True, coupler_mode='lambda', coupler_dim_in=channel // 2),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel, breadth))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, breadth
    ):
        super().__init__()

        blocks = [SwitchedConv(in_channel, channel, 3, breadth, padding=1, include_coupler=True, coupler_mode='lambda', coupler_dim_in=in_channel)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel, breadth))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    UpsampleConv(channel, channel // 2, breadth, 5, padding=2),
                    nn.ReLU(inplace=True),
                    UpsampleConv(
                        channel // 2, out_channel, breadth, 5, padding=2
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                UpsampleConv(channel, out_channel, breadth, 5, padding=2)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        codebook_dim=64,
        codebook_size=512,
        decay=0.99,
        breadth=4,
    ):
        super().__init__()

        self.breadth = breadth
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4, breadth=breadth)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2, breadth=breadth)
        self.quantize_conv_t = nn.Conv2d(channel, codebook_dim, 1)
        self.quantize_t = Quantize(codebook_dim, codebook_size)
        self.dec_t = Decoder(
            codebook_dim, codebook_dim, channel, n_res_block, n_res_channel, stride=2, breadth=breadth
        )
        self.quantize_conv_b = nn.Conv2d(codebook_dim + channel, codebook_dim, 1)
        self.quantize_b = Quantize(codebook_dim, codebook_size*2)
        self.upsample_t = UpsampleConv(
            codebook_dim, codebook_dim, breadth, 5, padding=2
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

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def save_attention_to_image_rgb(self, output_file, attention_out, attention_size, cmap_discrete_name='viridis'):
        from matplotlib import cm
        magnitude, indices = torch.topk(attention_out, 3, dim=1)
        indices = indices.cpu()
        colormap = cm.get_cmap(cmap_discrete_name, attention_size)
        img = torch.tensor(colormap(indices[:, 0, :, :].detach().numpy()))  # TODO: use other k's
        img = img.permute((0, 3, 1, 2))
        torchvision.utils.save_image(img, output_file)

    def visual_dbg(self, step, path):
        convs = [self.dec.blocks[-1].conv, self.dec_t.blocks[-1].conv, self.enc_b.blocks[-4], self.enc_t.blocks[-4]]
        for i, c in enumerate(convs):
            self.save_attention_to_image_rgb(os.path.join(path, "%i_selector_%i.png" % (step, i+1)), c.last_select, self.breadth)

    def encode(self, input):
        enc_b = checkpoint(self.enc_b, input)
        enc_t = checkpoint(self.enc_t, enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = checkpoint(self.dec_t, quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = checkpoint(self.quantize_conv_b, enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = checkpoint(self.dec, quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


def convert_weights(weights_file):
    sd = torch.load(weights_file)
    import models.vqvae.vqvae_no_conv_transpose as stdvq
    std_model = stdvq.VQVAE()
    std_model.load_state_dict(sd)
    nsd = convert_conv_net_state_dict_to_switched_conv(std_model, 4, ['quantize_conv_t', 'quantize_conv_b'])
    torch.save(nsd, "converted.pth")


@register_model
def register_vqvae_norm_switched_conv_lambda(opt_net, opt):
    kw = opt_get(opt_net, ['kwargs'], {})
    return VQVAE(**kw)


if __name__ == '__main__':
    #v = VQVAE()
    #print(v(torch.randn(1,3,128,128))[0].shape)
    convert_weights("../../../experiments/4000_generator.pth")
