from models.diffusion.fp16_util import convert_module_to_f32, convert_module_to_f16
from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import AttentionPool2d, AttentionBlock, ResBlock, TimestepEmbedSequential, \
    Downsample, Upsample
import torch
import torch.nn as nn

from models.gpt_voice.mini_encoder import AudioMiniEncoder, EmbeddingCombiner
from models.vqvae.vqvae import Quantize
from trainer.networks import register_model
import models.gpt_voice.my_dvae as mdvae
from utils.util import get_mask_from_lengths


class DiscreteEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 model_channels,
                 out_channels,
                 dropout,
                 scale):
        super().__init__()
        self.blocks = nn.Sequential(
            conv_nd(1, in_channels, model_channels, 3, padding=1),
            mdvae.ResBlock(model_channels, dropout, dims=1),
            Downsample(model_channels, use_conv=True, dims=1, out_channels=model_channels*2, factor=scale),
            mdvae.ResBlock(model_channels*2, dropout, dims=1),
            Downsample(model_channels*2, use_conv=True, dims=1, out_channels=model_channels*4, factor=scale),
            mdvae.ResBlock(model_channels*4, dropout, dims=1),
            AttentionBlock(model_channels*4, num_heads=4),
            mdvae.ResBlock(model_channels*4, dropout, out_channels=out_channels, dims=1),
        )

    def forward(self, spectrogram):
        return self.blocks(spectrogram)


class DiscreteDecoder(nn.Module):
    def __init__(self, in_channels, level_channels, scale):
        super().__init__()
        # Just raw upsampling, return a dict with each layer.
        self.init = conv_nd(1, in_channels, level_channels[0], kernel_size=3, padding=1)
        layers = []
        for i, lvl in enumerate(level_channels[:-1]):
            layers.append(nn.Sequential(normalization(lvl),
                                        nn.SiLU(lvl),
                                        Upsample(lvl, use_conv=True, dims=1, out_channels=level_channels[i+1], factor=scale)))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        y = self.init(x)
        outs = [y]
        for layer in self.layers:
            y = layer(y)
            outs.append(y)
        return outs


class DiffusionDVAE(nn.Module):
    def __init__(
            self,
            model_channels,
            num_res_blocks,
            in_channels=1,
            out_channels=2,  # mean and variance
            spectrogram_channels=80,
            spectrogram_conditioning_levels=[3,4,5],  # Levels at which spectrogram conditioning is applied to the waveform.
            dropout=0,
            channel_mult=(1, 2, 4, 8, 16, 32, 64),
            attention_resolutions=(16,32,64),
            conv_resample=True,
            dims=1,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            use_new_attention_order=False,
            kernel_size=5,
            quantize_dim=1024,
            num_discrete_codes=8192,
            scale_steps=4,
            conditioning_inputs_provided=True,
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
        self.spectrogram_conditioning_levels = spectrogram_conditioning_levels
        self.scale_steps = scale_steps

        self.encoder = DiscreteEncoder(spectrogram_channels, model_channels*4, quantize_dim, dropout, scale_steps)
        self.quantizer = Quantize(quantize_dim, num_discrete_codes)
        # For recording codebook usage.
        self.codes = torch.zeros((131072,), dtype=torch.long)
        self.code_ind = 0
        self.internal_step = 0
        decoder_channels = [model_channels * channel_mult[s-1] for s in spectrogram_conditioning_levels]
        self.decoder = DiscreteDecoder(quantize_dim, decoder_channels[::-1], scale_steps)

        padding = 1 if kernel_size == 3 else 2

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.conditioning_enabled = conditioning_inputs_provided
        if conditioning_inputs_provided:
            self.contextual_embedder = AudioMiniEncoder(self.spectrogram_channels, time_embed_dim)
            self.query_gen = AudioMiniEncoder(decoder_channels[0], time_embed_dim)
            self.embedding_combiner = EmbeddingCombiner(time_embed_dim)

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
        self.convergence_convs = nn.ModuleList([])
        for level, mult in enumerate(channel_mult):
            if level in spectrogram_conditioning_levels:
                self.convergence_convs.append(conv_nd(dims, ch*2, ch, 1))

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
                        Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, factor=scale_steps
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
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, factor=scale_steps)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, kernel_size, padding=padding)),
        )

    def _decode_continouous(self, x, timesteps, embeddings, conditioning_inputs, num_conditioning_signals):
        if self.conditioning_enabled:
            assert conditioning_inputs is not None

        spec_hs = self.decoder(embeddings)[::-1]
        # Shape the spectrogram correctly. There is no guarantee it fits (though I probably should add an assertion here to make sure the resizing isn't too wacky.)
        spec_hs = [nn.functional.interpolate(sh, size=(x.shape[-1]//self.scale_steps**self.spectrogram_conditioning_levels[i],), mode='nearest') for i, sh in enumerate(spec_hs)]
        convergence_fns = list(self.convergence_convs)

        # Timestep embeddings and conditioning signals are combined using a small transformer.
        hs = []
        emb1 = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.conditioning_enabled:
            mask = get_mask_from_lengths(num_conditioning_signals+1, conditioning_inputs.shape[1]+1)  # +1 to account for the timestep embeddings we'll add.
            emb2 = torch.stack([self.contextual_embedder(ci.squeeze(1)) for ci in list(torch.chunk(conditioning_inputs, conditioning_inputs.shape[1], dim=1))], dim=1)
            emb = torch.cat([emb1.unsqueeze(1), emb2], dim=1)
            emb = self.embedding_combiner(emb, mask, self.query_gen(spec_hs[0]))
        else:
            emb = emb1

        # The rest is the diffusion vocoder, built as a standard U-net. spec_h is gradually fed into the encoder.
        next_spec = spec_hs.pop(0)
        next_convergence_fn = convergence_fns.pop(0)
        h = x.type(self.dtype)
        for k, module in enumerate(self.input_blocks):
            h = module(h, emb)
            if next_spec is not None and h.shape[-1] == next_spec.shape[-1]:
                h = torch.cat([h, next_spec], dim=1)
                h = next_convergence_fn(h)
                if len(spec_hs) > 0:
                    next_spec = spec_hs.pop(0)
                    next_convergence_fn = convergence_fns.pop(0)
                else:
                    next_spec = None
            hs.append(h)
        assert len(spec_hs) == 0
        assert len(convergence_fns) == 0
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def decode(self, x, timesteps, codes, conditioning_inputs=None, num_conditioning_signals=None):
        assert x.shape[-1] % 4096 == 0  # This model operates at base//4096 at it's bottom levels, thus this requirement.
        embeddings = self.quantizer.embed_code(codes).permute((0,2,1))
        return self._decode_continouous(x, timesteps, embeddings, conditioning_inputs, num_conditioning_signals)

    def forward(self, x, timesteps, spectrogram, conditioning_inputs=None, num_conditioning_signals=None):
        assert x.shape[-1] % 4096 == 0  # This model operates at base//4096 at it's bottom levels, thus this requirement.

        # Compute DVAE portion first.
        spec_logits = self.encoder(spectrogram).permute((0,2,1))
        sampled, commitment_loss, codes = self.quantizer(spec_logits)
        if self.training:
            # Compute from softmax outputs to preserve gradients.
            embeddings = sampled.permute((0,2,1))
        else:
            # Compute from codes only.
            embeddings = self.quantizer.embed_code(codes).permute((0,2,1))
        return self._decode_continouous(x, timesteps, embeddings, conditioning_inputs, num_conditioning_signals), commitment_loss


@register_model
def register_unet_diffusion_dvae(opt_net, opt):
    return DiffusionDVAE(**opt_net['kwargs'])


# Test for ~4 second audio clip at 22050Hz
if __name__ == '__main__':
    clip = torch.randn(4, 1, 81920)
    spec = torch.randn(4, 80, 416)
    cond = torch.randn(4, 5, 80, 200)
    num_cond = torch.tensor([2,4,5,3], dtype=torch.long)
    ts = torch.LongTensor([432, 234, 100, 555])
    model = DiffusionDVAE(32, 2)
    print(model(clip, ts, spec, cond, num_cond)[0].shape)
