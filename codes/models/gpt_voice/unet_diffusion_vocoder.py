from models.diffusion.fp16_util import convert_module_to_f32, convert_module_to_f16
from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import AttentionPool2d, AttentionBlock, ResBlock, TimestepEmbedSequential, \
    Downsample, Upsample
import torch
import torch.nn as nn

from trainer.networks import register_model


class DiffusionVocoder(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    Customized to be conditioned on a spectrogram prior.

    :param in_channels: channels in the input Tensor.
    :param spectrogram_channels: channels in the conditioning spectrogram.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            model_channels,
            num_res_blocks,
            in_channels=1,
            out_channels=2,  # mean and variance
            spectrogram_channels=80,
            spectrogram_conditioning_level=3,  # Level at which spectrogram conditioning is applied to the waveform.
            dropout=0,
            # 106496 -> 26624 -> 6656 -> 16664 -> 416 -> 104 -> 26  for ~5secs@22050Hz
            channel_mult=(1, 2, 4, 8, 16, 32, 64),
            attention_resolutions=(16,32,64),
            conv_resample=True,
            dims=1,
            num_classes=None,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            kernel_size=5,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.spectrogram_channels = spectrogram_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dims = dims

        padding = 1 if kernel_size == 3 else 2

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, kernel_size, padding=padding)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        spec_chs = channel_mult[spectrogram_conditioning_level] * model_channels
        self.spectrogram_conditioner = nn.Sequential(
            conv_nd(dims, self.spectrogram_channels, spec_chs, 1),
            normalization(spec_chs),
            nn.SiLU(),
            conv_nd(dims, spec_chs, spec_chs, 1)
        )
        self.convergence_conv = nn.Sequential(
            normalization(spec_chs*2),
            nn.SiLU(),
            conv_nd(dims, spec_chs*2, spec_chs*2, 1)
        )

        for level, mult in enumerate(channel_mult):
            if level == spectrogram_conditioning_level+1:
                ch *= 2  # At this level, the spectrogram is concatenated onto the input.

            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        kernel_size=kernel_size,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            kernel_size=kernel_size,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
            if level == spectrogram_conditioning_level:
                self.input_block_injection_point = len(self.input_blocks)-1
                input_block_chans[-1] *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                kernel_size=kernel_size,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                kernel_size=kernel_size,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        kernel_size=kernel_size,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            kernel_size=kernel_size,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, kernel_size, padding=padding)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, spectrogram):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert x.shape[-1] % 4096 == 0  # This model operates at base//4096 at it's bottom levels, thus this requirement.
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        conditioning = self.spectrogram_conditioner(spectrogram)

        h = x.type(self.dtype)
        for k, module in enumerate(self.input_blocks):
            h = module(h, emb)
            if k == self.input_block_injection_point:
                cond = nn.functional.interpolate(conditioning, size=h.shape[-self.dims:], mode='nearest')
                h = torch.cat([h, cond], dim=1)
                h = self.convergence_conv(h)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


@register_model
def register_unet_diffusion_vocoder(opt_net, opt):
    return DiffusionVocoder(**opt_net['kwargs'])


# Test for ~4 second audio clip at 22050Hz
if __name__ == '__main__':
    clip = torch.randn(1, 1, 81920)
    spec = torch.randn(1, 80, 416)
    ts = torch.LongTensor([555])
    model = DiffusionVocoder(16, 2)
    print(model(clip, ts, spec).shape)
