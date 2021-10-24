from models.diffusion.fp16_util import convert_module_to_f32, convert_module_to_f16
from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import AttentionPool2d, AttentionBlock, ResBlock, TimestepEmbedSequential, \
    Downsample, Upsample
import torch
import torch.nn as nn

from models.gpt_voice.mini_encoder import AudioMiniEncoder, EmbeddingCombiner
from trainer.networks import register_model
from utils.util import get_mask_from_lengths


class DiscreteSpectrogramConditioningBlock(nn.Module):
    def __init__(self, dvae_channels, channels):
        super().__init__()
        self.intg = nn.Sequential(nn.Conv1d(dvae_channels, channels, kernel_size=1),
                                  normalization(channels),
                                  nn.SiLU(),
                                  nn.Conv1d(channels, channels, kernel_size=3))

    """
    Embeds the given codes and concatenates them onto x. Return shape is the same as x.shape.
    
    :param x: bxcxS waveform latent
    :param codes: bxN discrete codes, N <= S
    """
    def forward(self, x, dvae_in):
        b, c, S = x.shape
        _, q, N = dvae_in.shape
        emb = self.intg(dvae_in)
        emb = nn.functional.interpolate(emb, size=(S,), mode='nearest')
        return torch.cat([x, emb], dim=1)


class DiffusionVocoderWithRef(nn.Module):
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
            in_channels=1,
            out_channels=2,  # mean and variance
            discrete_codes=512,
            dropout=0,
            # res           1, 2, 4, 8,16,32,64,128,256,512, 1K, 2K
            channel_mult=  (1,1.5,2, 3, 4, 6, 8, 12, 16, 24, 32, 48),
            num_res_blocks=(1, 1, 1, 1, 1, 2, 2, 2,   2,  2,  2,  2),
            # spec_cond:    1, 0, 0, 1, 0, 0, 1, 0,   0,  1,  0,  0)
            # attn:         0, 0, 0, 0, 0, 0, 0, 0,   0,  1,  1,  1
            spectrogram_conditioning_resolutions=(512,),
            attention_resolutions=(512,1024,2048),
            conv_resample=True,
            dims=1,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            kernel_size=3,
            scale_factor=2,
            conditioning_inputs_provided=True,
            conditioning_input_dim=80,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
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

        self.conditioning_enabled = conditioning_inputs_provided
        if conditioning_inputs_provided:
            self.contextual_embedder = AudioMiniEncoder(conditioning_input_dim, time_embed_dim)
            self.query_gen = AudioMiniEncoder(in_channels, time_embed_dim, base_channels=32, depth=6, resnet_blocks=1,
                                              attn_blocks=2, num_attn_heads=2, dropout=dropout, downsample_factor=4, kernel_size=5)
            self.embedding_combiner = EmbeddingCombiner(time_embed_dim, attn_blocks=1)

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

        for level, (mult, num_blocks) in enumerate(zip(channel_mult, num_res_blocks)):
            if ds in spectrogram_conditioning_resolutions:
                self.input_blocks.append(DiscreteSpectrogramConditioningBlock(discrete_codes, ch))
                ch *= 2

            for _ in range(num_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        kernel_size=kernel_size,
                    )
                ]
                ch = int(mult * model_channels)
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
                            ch, conv_resample, dims=dims, out_channels=out_ch, factor=scale_factor
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

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
        for level, (mult, num_blocks) in list(enumerate(zip(channel_mult, num_res_blocks)))[::-1]:
            for i in range(num_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        kernel_size=kernel_size,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_blocks:
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
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, factor=scale_factor)
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

    def forward(self, x, timesteps, spectrogram, conditioning_inputs=None, num_conditioning_signals=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert x.shape[-1] % 4096 == 0  # This model operates at base//4096 at it's bottom levels, thus this requirement.
        if self.conditioning_enabled:
            assert conditioning_inputs is not None
            assert num_conditioning_signals is not None

        hs = []
        emb1 = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.conditioning_enabled:
            emb2 = torch.stack([self.contextual_embedder(ci.squeeze(1)) for ci in list(torch.chunk(conditioning_inputs, conditioning_inputs.shape[1], dim=1))], dim=1)
            emb2 = self.embedding_combiner(emb2, None, self.query_gen(x))
            emb = emb1 + emb2
        else:
            emb = emb1

        h = x.type(self.dtype)
        for k, module in enumerate(self.input_blocks):
            if isinstance(module, DiscreteSpectrogramConditioningBlock):
                h = module(h, spectrogram)
            else:
                h = module(h, emb)
                hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


@register_model
def register_unet_diffusion_vocoder_with_ref(opt_net, opt):
    return DiffusionVocoderWithRef(**opt_net['kwargs'])


# Test for ~4 second audio clip at 22050Hz
if __name__ == '__main__':
    clip = torch.randn(2, 1, 40960)
    #spec = torch.randint(8192, (2, 40,))
    spec = torch.randn(2,512,160)
    cond = torch.randn(2, 3, 80, 173)
    ts = torch.LongTensor([555, 556])
    model = DiffusionVocoderWithRef(32, conditioning_inputs_provided=True)
    print(model(clip, ts, spec, cond, 3).shape)
