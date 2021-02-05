import os
from time import time

import torch
import torchvision
from torch import nn
from tqdm import tqdm

from models.switched_conv.switched_conv_hard_routing import SwitchedConvHardRouting, \
    convert_conv_net_state_dict_to_switched_conv
from models.vqvae.vqvae import ResBlock, Quantize
from trainer.networks import register_model
from utils.util import checkpoint, opt_get


# Upsamples and blurs (similar to StyleGAN). Replaces ConvTranspose2D from the original paper.
class UpsampleConv(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, padding, cfg):
        super().__init__()
        self.conv = SwitchedConvHardRouting(in_filters, out_filters, kernel_size, breadth=cfg['breadth'], include_coupler=True,
                                            coupler_mode=cfg['mode'], coupler_dim_in=in_filters, dropout_rate=cfg['dropout'], hard_en=cfg['hard_enabled'])

    def forward(self, x):
        up = torch.nn.functional.interpolate(x, scale_factor=2)
        return self.conv(up)


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride, cfg):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 5, stride=2, padding=2),
                nn.LeakyReLU(inplace=True),
                SwitchedConvHardRouting(channel // 2, channel, 5, breadth=cfg['breadth'], stride=2, include_coupler=True,
                                        coupler_mode=cfg['mode'], coupler_dim_in=channel // 2, dropout_rate=cfg['dropout'], hard_en=cfg['hard_enabled']),
                nn.LeakyReLU(inplace=True),
                SwitchedConvHardRouting(channel, channel, 3, breadth=cfg['breadth'], include_coupler=True, coupler_mode=cfg['mode'],
                                        coupler_dim_in=channel, dropout_rate=cfg['dropout'], hard_en=cfg['hard_enabled']),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 5, stride=2, padding=2),
                nn.LeakyReLU(inplace=True),
                SwitchedConvHardRouting(channel // 2, channel, 3, breadth=cfg['breadth'], include_coupler=True, coupler_mode=cfg['mode'],
                                        coupler_dim_in=channel // 2, dropout_rate=cfg['dropout'], hard_en=cfg['hard_enabled']),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.LeakyReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, cfg
    ):
        super().__init__()

        blocks = [SwitchedConvHardRouting(in_channel, channel, 3, breadth=cfg['breadth'], include_coupler=True, coupler_mode=cfg['mode'],
                                          coupler_dim_in=in_channel, dropout_rate=cfg['dropout'], hard_en=cfg['hard_enabled'])]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.LeakyReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    UpsampleConv(channel, channel // 2, 5, padding=2, cfg=cfg),
                    nn.LeakyReLU(inplace=True),
                    UpsampleConv(
                        channel // 2, out_channel, 5, padding=2, cfg=cfg
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                UpsampleConv(channel, out_channel, 5, padding=2, cfg=cfg)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE3HardSwitch(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        codebook_dim=64,
        codebook_size=512,
        decay=0.99,
        cfg={'mode':'standard', 'breadth':16, 'hard_enabled': True, 'dropout': 0.4}
    ):
        super().__init__()

        self.cfg = cfg
        self.initial_conv = nn.Sequential(*[nn.Conv2d(in_channel, 32, 3, padding=1),
                                           nn.LeakyReLU(inplace=True)])
        self.enc_b = Encoder(32, channel, n_res_block, n_res_channel, stride=4, cfg=cfg)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2, cfg=cfg)
        self.quantize_conv_t = nn.Conv2d(channel, codebook_dim, 1)
        self.quantize_t = Quantize(codebook_dim, codebook_size)
        self.dec_t = Decoder(
            codebook_dim, codebook_dim, channel, n_res_block, n_res_channel, stride=2, cfg=cfg
        )
        self.quantize_conv_b = nn.Conv2d(codebook_dim + channel, codebook_dim, 1)
        self.quantize_b = Quantize(codebook_dim, codebook_size)
        self.upsample_t = UpsampleConv(
            codebook_dim, codebook_dim, 5, padding=2, cfg=cfg
        )
        self.dec = Decoder(
            codebook_dim + codebook_dim,
            32,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
            cfg=cfg
        )
        self.final_conv = nn.Conv2d(32, in_channel, 3, padding=1)

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
            self.save_attention_to_image_rgb(os.path.join(path, "%i_selector_%i.png" % (step, i+1)), c.last_select, 16)

    def get_debug_values(self, step, __):
        switched_convs = [('enc_b_blk2', self.enc_b.blocks[2]),
                          ('enc_b_blk4', self.enc_b.blocks[4]),
                          ('enc_t_blk2', self.enc_t.blocks[2]),
                          ('dec_t_blk0', self.dec_t.blocks[0]),
                          ('dec_t_blk-1', self.dec_t.blocks[-1].conv),
                          ('dec_blk0', self.dec.blocks[0]),
                          ('dec_blk-1', self.dec.blocks[-1].conv),
                          ('dec_blk-3', self.dec.blocks[-3].conv)]
        logs = {}
        for name, swc in switched_convs:
            logs[f'{name}_histogram_switch_usage'] = swc.latest_masks
        return logs

    def encode(self, input):
        fea = self.initial_conv(input)
        enc_b = checkpoint(self.enc_b, fea)
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
        dec = checkpoint(self.final_conv, dec)

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
    from models.vqvae.vqvae_3 import VQVAE3
    std_model = VQVAE3()
    std_model.load_state_dict(sd)
    nsd = convert_conv_net_state_dict_to_switched_conv(std_model, 16, ['quantize_conv_t', 'quantize_conv_b',
                                                                      'enc_b.blocks.0', 'enc_t.blocks.0',
                                                                      'conv.1', 'conv.3', 'initial_conv', 'final_conv'])
    torch.save(nsd, "converted.pth")


@register_model
def register_vqvae3_hard_switch(opt_net, opt):
    kw = opt_get(opt_net, ['kwargs'], {})
    return VQVAE3HardSwitch(**kw)


def performance_test():
    cfg = {
        'mode': 'lambda',
        'breadth': 16,
        'hard_enabled': True,
        'dropout': 0.4
    }
    net = VQVAE3HardSwitch(cfg=cfg).to('cuda')
    loss = nn.L1Loss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    started = time()
    for j in tqdm(range(10)):
        inp = torch.rand((8, 3, 256, 256), device='cuda')
        res = net(inp)[0]
        l = loss(res, inp)
        l.backward()
        opt.step()
        net.zero_grad()
    print("Elapsed: ", (time()-started))


if __name__ == '__main__':
    #v = VQVAE3HardSwitch()
    #print(v(torch.randn(1,3,128,128))[0].shape)
    #convert_weights("../../../experiments/vqvae_base.pth")
    performance_test()
