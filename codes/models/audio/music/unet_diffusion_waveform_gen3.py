import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import TimestepEmbedSequential, \
    Downsample, Upsample, TimestepBlock
from scripts.audio.gen.use_diffuse_tts import ceil_multiple
from trainer.networks import register_model
from utils.util import checkpoint, print_network


def is_sequence(t):
    return t.dtype == torch.long


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        kernel_size=3,
        efficient_config=False,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = {1: 0, 3: 1, 5: 2}[kernel_size]
        eff_kernel = 1 if efficient_config else 3
        eff_padding = 0 if efficient_config else 1

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, eff_kernel, padding=eff_padding),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, eff_kernel, padding=eff_padding)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, x, emb
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class StackedResidualBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout):
        super().__init__()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * channels,
            ),
        )

        gc = channels // 4
        self.initial_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        for i in range(5):
            out_channels = channels if i == 4 else gc
            self.add_module(
                f'conv{i + 1}',
                nn.Conv1d(channels + i * gc, out_channels, 3, 1, 1))
            if i != 4:
                self.add_module(f'gn{i+1}', nn.GroupNorm(num_groups=8, num_channels=out_channels))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        zero_module(self.conv5)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x, emb):
        return checkpoint(self.forward_, x, emb)

    def forward_(self, x, emb):
        emb_out = self.emb_layers(emb)
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        x0 = self.initial_norm(x) * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        x1 = self.lrelu(self.gn1(self.conv1(x0)))
        x2 = self.lrelu(self.gn2(self.conv2(torch.cat((x, x1), 1))))
        x3 = self.lrelu(self.gn3(self.conv3(torch.cat((x, x1, x2), 1))))
        x4 = self.lrelu(self.gn4(self.conv4(torch.cat((x, x1, x2, x3), 1))))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = self.drop(x5)

        return x5 + x


class DiffusionWaveformGen(nn.Module):
    """
    The full UNet model with residual blocks and timestep embedding.

    Customized to be conditioned on an aligned prior derived from a autoregressive
    GPT-style model.

    :param in_channels: channels in the input Tensor.
    :param in_latent_channels: channels from the input latent.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    """

    def __init__(
            self,
            model_channels=512,
            in_channels=64,
            in_mel_channels=256,
            conditioning_dim_factor=2,
            out_channels=128,  # mean and variance
            dropout=0,
            channel_mult=  (1,1.5,2),
            num_res_blocks=(1,1,0),
            token_conditioning_resolutions=(1,4),
            mid_resnet_depth=10,
            use_fp16=False,
            time_embed_dim_multiplier=1,
            # Parameters for regularization.
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.alignment_size = 2 ** (len(channel_mult)+1)
        self.in_mel_channels = in_mel_channels

        time_embed_dim = model_channels * time_embed_dim_multiplier
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        conditioning_dim = model_channels * conditioning_dim_factor
        # Either code_converter or latent_converter is used, depending on what type of conditioning data is fed.
        # This model is meant to be able to be trained on both for efficiency purposes - it is far less computationally
        # complex to generate tokens, while generating latents will normally mean propagating through a deep autoregressive
        # transformer network.
        self.mel_converter = nn.Conv1d(in_mel_channels, conditioning_dim, 3, padding=1)
        self.unconditioned_embedding = nn.Parameter(torch.randn(1,conditioning_dim,1))

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(1, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        token_conditioning_blocks = []
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, (mult, num_blocks) in enumerate(zip(channel_mult, num_res_blocks)):
            if ds in token_conditioning_resolutions:
                token_conditioning_block = nn.Conv1d(conditioning_dim, ch, 1)
                token_conditioning_block.weight.data *= .02
                self.input_blocks.append(token_conditioning_block)
                token_conditioning_blocks.append(token_conditioning_block)

            for _ in range(num_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=1,
                        kernel_size=3,
                        use_scale_shift_norm=True,
                    )
                ]
                ch = int(mult * model_channels)
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(
                            ch, True, dims=1, out_channels=out_ch, factor=2, ksize=3, pad=1
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(nn.Conv1d(ch+conditioning_dim, ch, kernel_size=1),
                                                    *[StackedResidualBlock(ch, time_embed_dim, dropout) for _ in range(mid_resnet_depth)])
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, (mult, num_blocks) in list(enumerate(zip(channel_mult, num_res_blocks)))[::-1]:
            for i in range(num_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=1,
                        kernel_size=3,
                        use_scale_shift_norm=True,
                    )
                ]
                ch = int(model_channels * mult)
                if level and i == num_blocks:
                    out_ch = ch
                    layers.append(
                        Upsample(ch, True, dims=1, out_channels=out_ch, factor=2)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(1, model_channels, out_channels, 3, padding=1)),
        )

    def get_grad_norm_parameter_groups(self):
        groups = {
            'input_blocks': list(self.input_blocks.parameters()),
            'output_blocks': list(self.output_blocks.parameters()),
            'middle_rrdb': list(self.middle_block.parameters()),
        }
        return groups

    def fix_alignment(self, x, aligned_conditioning):
        """
        The UNet requires that the input <x> is a certain multiple of 2, defined by the UNet depth. Enforce this by
        padding both <x> and <aligned_conditioning> before forward propagation and removing the padding before returning.
        """
        cm = ceil_multiple(x.shape[-1], self.alignment_size)
        if cm != 0:
            pc = (cm-x.shape[-1])/x.shape[-1]
            x = F.pad(x, (0,cm-x.shape[-1]))
            aligned_conditioning = F.pad(aligned_conditioning, (0,int(pc*aligned_conditioning.shape[-1])))
        return x, aligned_conditioning

    def forward(self, x, timesteps, codes, conditioning_free=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param codes: an aligned latent or sequence of tokens providing useful data about the sample to be produced.
        :param conditioning_free: When set, all conditioning inputs (including tokens and conditioning_input) will not be considered.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # Fix input size to the proper multiple of 2 so we don't get alignment errors going down and back up the U-net.
        orig_x_shape = x.shape[-1]
        x, codes = self.fix_alignment(x, codes)


        hs = []
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Note: this block does not need to repeated on inference, since it is not timestep-dependent.
        if conditioning_free:
            code_emb = self.unconditioned_embedding.repeat(x.shape[0], 1, 1)
        else:
            code_emb = self.mel_converter(codes)

        time_emb = time_emb.float()
        h = x
        for k, module in enumerate(self.input_blocks):
            if isinstance(module, nn.Conv1d):
                h_tok = F.interpolate(module(code_emb), size=(h.shape[-1]), mode='nearest')
                h = h + h_tok
            else:
                h = module(h, time_emb)
                hs.append(h)
        h = torch.cat([h, F.interpolate(code_emb, size=(h.shape[-1]), mode='nearest')], dim=1)
        h = self.middle_block(h, time_emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, time_emb)

        out = self.out(h)

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        params = [self.unconditioned_embedding]
        for p in params:
            extraneous_addition = extraneous_addition + p.mean()
        out = out + extraneous_addition * 0

        return out[:, :, :orig_x_shape]


@register_model
def register_unet_diffusion_waveform_gen3(opt_net, opt):
    return DiffusionWaveformGen(**opt_net['kwargs'])


if __name__ == '__main__':
    clip = torch.randn(2, 4, 880)
    aligned_sequence = torch.randn(2,256,220)
    ts = torch.LongTensor([600, 600])
    model = DiffusionWaveformGen(in_channels=4, out_channels=8, model_channels=64, in_mel_channels=256,
                                 channel_mult=[1,2,4,6,8,16], num_res_blocks=[2,2,2,1,1,0], mid_resnet_depth=24,
                                 conditioning_dim_factor=8,
                                 token_conditioning_resolutions=[4,16], dropout=.1, time_embed_dim_multiplier=4)
    # Test with sequence aligned conditioning
    o = model(clip, ts, aligned_sequence)
    print_network(model)

