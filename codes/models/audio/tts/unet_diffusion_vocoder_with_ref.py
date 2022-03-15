from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import AttentionBlock, ResBlock, TimestepEmbedSequential, \
    Downsample, Upsample
import torch
import torch.nn as nn

from models.audio.tts.mini_encoder import AudioMiniEncoder
from trainer.networks import register_model


class DiscreteSpectrogramConditioningBlock(nn.Module):
    def __init__(self, dvae_channels, channels, level):
        super().__init__()
        self.intg = nn.Sequential(nn.Conv1d(dvae_channels, channels, kernel_size=1),
                                  normalization(channels),
                                  nn.SiLU(),
                                  nn.Conv1d(channels, channels, kernel_size=3))
        self.level = level

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
            time_embed_dim_multiplier=4,
            freeze_layers_below=None,  # powers of 2; ex: 1,2,4,8,16,32,etc..
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

        time_embed_dim = model_channels * time_embed_dim_multiplier
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.conditioning_enabled = conditioning_inputs_provided
        if conditioning_inputs_provided:
            self.contextual_embedder = AudioMiniEncoder(in_channels, time_embed_dim, base_channels=32, depth=6, resnet_blocks=1,
                             attn_blocks=2, num_attn_heads=2, dropout=dropout, downsample_factor=4, kernel_size=5)

        seqlyr = TimestepEmbedSequential(
            conv_nd(dims, in_channels, model_channels, kernel_size, padding=padding)
        )
        seqlyr.level = 0
        self.input_blocks = nn.ModuleList([seqlyr])
        spectrogram_blocks = []
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, (mult, num_blocks) in enumerate(zip(channel_mult, num_res_blocks)):
            if ds in spectrogram_conditioning_resolutions:
                spec_cond_block = DiscreteSpectrogramConditioningBlock(discrete_codes, ch, 2 ** level)
                self.input_blocks.append(spec_cond_block)
                spectrogram_blocks.append(spec_cond_block)
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
                layer = TimestepEmbedSequential(*layers)
                layer.level = 2 ** level
                self.input_blocks.append(layer)
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                upblk = TimestepEmbedSequential(
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
                upblk.level = 2 ** level
                self.input_blocks.append(upblk)
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
                layer = TimestepEmbedSequential(*layers)
                layer.level = 2 ** level
                self.output_blocks.append(layer)
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, kernel_size, padding=padding)),
        )

        if freeze_layers_below is not None:
            # Freeze all parameters first.
            for p in self.parameters():
                p.DO_NOT_TRAIN = True
                p.requires_grad = False
            # Now un-freeze the modules we actually want to train.
            unfrozen_modules = [self.out]
            for blk in self.input_blocks:
                if blk.level <= freeze_layers_below:
                    unfrozen_modules.append(blk)
            last_frozen_output_block = None
            for blk in self.output_blocks:
                if blk.level <= freeze_layers_below:
                    unfrozen_modules.append(blk)
                else:
                    last_frozen_output_block = blk
            # And finally, the last upsample block in output blocks.
            unfrozen_modules.append(last_frozen_output_block[1])
            unfrozen_params = 0
            for m in unfrozen_modules:
                for p in m.parameters():
                    del p.DO_NOT_TRAIN
                    p.requires_grad = True
                    unfrozen_params += 1
            print(f"freeze_layers_below specified. Training a total of {unfrozen_params} parameters.")

    def forward(self, x, timesteps, spectrogram, conditioning_input=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert x.shape[-1] % 2048 == 0  # This model operates at base//2048 at it's bottom levels, thus this requirement.
        if self.conditioning_enabled:
            assert conditioning_input is not None

        hs = []
        emb1 = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.conditioning_enabled:
            emb2 = self.contextual_embedder(conditioning_input)
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


def move_all_layers_down(pretrained_path, output_path, layers_to_be_added=3):
    # layers_to_be_added should be=num_res_blocks+1+[1if spectrogram_conditioning_resolutions;else0]
    sd = torch.load(pretrained_path)
    out = sd.copy()
    replaced = []
    for n, p in sd.items():
        if n.startswith('input_blocks.') and not n.startswith('input_blocks.0.'):
            if n not in replaced:
                del out[n]
            components = n.split('.')
            components[1] = str(int(components[1]) + layers_to_be_added)
            new_name = '.'.join(components)
            out[new_name] = p
            replaced.append(new_name)
    torch.save(out, output_path)


@register_model
def register_unet_diffusion_vocoder_with_ref(opt_net, opt):
    return DiffusionVocoderWithRef(**opt_net['kwargs'])


# Test for ~4 second audio clip at 22050Hz
if __name__ == '__main__':
    path = 'X:\\dlas\\experiments\\train_diffusion_vocoder_with_cond_new_dvae_full\\models\\6100_generator_ema.pth'
    move_all_layers_down(path, 'diffuse_new_lyr.pth', layers_to_be_added=2)

    clip = torch.randn(2, 1, 40960)
    spec = torch.randn(2,80,160)
    cond = torch.randn(2, 1, 40960)
    ts = torch.LongTensor([555, 556])
    model = DiffusionVocoderWithRef(model_channels=128, channel_mult=[1,1,1.5,2, 3, 4, 6, 8, 8,   8,  8 ],
                                    num_res_blocks=[1,2, 2, 2, 2, 2, 2, 2, 2,   1,  1 ], spectrogram_conditioning_resolutions=[2,512],
                                    dropout=.05, attention_resolutions=[512,1024], num_heads=4, kernel_size=3, scale_factor=2,
                                    conditioning_inputs_provided=True, conditioning_input_dim=80, time_embed_dim_multiplier=4,
                                    discrete_codes=80, freeze_layers_below=1)
    loading_errors = model.load_state_dict(torch.load('diffuse_new_lyr.pth'), strict=False)
    new_params = loading_errors.missing_keys
    new_params_trained = []
    existing_params_trained = []
    for n,p in model.named_parameters():
        if not hasattr(p, 'DO_NOT_TRAIN'):
            if n in new_params:
                new_params_trained.append(n)
            else:
                existing_params_trained.append(n)
    for n in new_params:
        if n not in new_params_trained:
            print(f"{n} is a new parameter, but it is not marked as trainable.")

    print(model(clip, ts, spec, cond).shape)
