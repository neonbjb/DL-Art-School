import operator
from collections import OrderedDict

from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import AttentionPool2d, AttentionBlock, TimestepEmbedSequential, \
    Downsample, Upsample, TimestepBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gpt_voice.mini_encoder import AudioMiniEncoder, EmbeddingCombiner
from scripts.audio.gen.use_diffuse_tts import ceil_multiple
from trainer.networks import register_model
from utils.util import checkpoint


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        kernel_size=3,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        padding = 1 if kernel_size == 3 else 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 1, padding=0),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                self.out_channels,
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
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

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
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class DiffusionTts(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    Customized to be conditioned on an aligned token prior.

    :param in_channels: channels in the input Tensor.
    :param num_tokens: number of tokens (e.g. characters) which can be provided.
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
            num_tokens=32,
            out_channels=2,  # mean and variance
            dropout=0,
            # res           1, 2, 4, 8,16,32,64,128,256,512, 1K, 2K
            channel_mult=  (1,1.5,2, 3, 4, 6, 8, 12, 16, 24, 32, 48),
            num_res_blocks=(1, 1, 1, 1, 1, 2, 2, 2,   2,  2,  2,  2),
            # spec_cond:    1, 0, 0, 1, 0, 0, 1, 0,   0,  1,  0,  0)
            # attn:         0, 0, 0, 0, 0, 0, 0, 0,   0,  1,  1,  1
            token_conditioning_resolutions=(1,16,),
            attention_resolutions=(512,1024,2048),
            conv_resample=True,
            dims=1,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            kernel_size=3,
            scale_factor=2,
            conditioning_inputs_provided=True,
            time_embed_dim_multiplier=4,
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

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, kernel_size, padding=padding)
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
                token_conditioning_block = nn.Embedding(num_tokens, ch)
                token_conditioning_block.weight.data.normal_(mean=0.0, std=.02)
                self.input_blocks.append(token_conditioning_block)
                token_conditioning_blocks.append(token_conditioning_block)

            for _ in range(num_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
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
                            ch, conv_resample, dims=dims, out_channels=out_ch, factor=scale_factor, ksize=1, pad=0
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
                kernel_size=kernel_size,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
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
                        )
                    )
                if level and i == num_blocks:
                    out_ch = ch
                    layers.append(
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, factor=scale_factor)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, kernel_size, padding=padding)),
        )


    def forward(self, x, timesteps, tokens, conditioning_input=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param tokens: an aligned text input.
        :return: an [N x C x ...] Tensor of outputs.
        """
        orig_x_shape = x.shape[-1]
        cm = ceil_multiple(x.shape[-1], 2048)
        if cm != 0:
            pc = (cm-x.shape[-1])/x.shape[-1]
            x = F.pad(x, (0,cm-x.shape[-1]))
            tokens = F.pad(tokens, (0,int(pc*tokens.shape[-1])))
        if self.conditioning_enabled:
            assert conditioning_input is not None

        hs = []
        emb1 = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.conditioning_enabled:
            actual_cond = self.contextual_embedder(conditioning_input)
            emb = emb1 + actual_cond
        else:
            emb = emb1

        h = x.type(self.dtype)
        for k, module in enumerate(self.input_blocks):
            if isinstance(module, nn.Embedding):
                h_tok = F.interpolate(module(tokens).permute(0,2,1), size=(h.shape[-1]), mode='nearest')
                h = h + h_tok
            else:
                h = module(h, emb)
                hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        return out[:, :, :orig_x_shape]

    def benchmark(self, x, timesteps, tokens, conditioning_input):
        profile = OrderedDict()
        params = OrderedDict()
        hs = []
        emb1 = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        from torchprofile import profile_macs
        profile['contextual_embedder'] = profile_macs(self.contextual_embedder, args=(conditioning_input,))
        params['contextual_embedder'] = sum(p.numel() for p in self.contextual_embedder.parameters())
        emb2 = self.contextual_embedder(conditioning_input)
        emb = emb1 + emb2

        h = x.type(self.dtype)
        for k, module in enumerate(self.input_blocks):
            if isinstance(module, nn.Embedding):
                h_tok = F.interpolate(module(tokens).permute(0,2,1), size=(h.shape[-1]), mode='nearest')
                h = h + h_tok
            else:
                profile[f'in_{k}'] = profile_macs(module, args=(h,emb))
                params[f'in_{k}'] = sum(p.numel() for p in module.parameters())
                h = module(h, emb)
                hs.append(h)
        profile['middle'] = profile_macs(self.middle_block, args=(h,emb))
        params['middle'] = sum(p.numel() for p in self.middle_block.parameters())
        h = self.middle_block(h, emb)
        for k, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            profile[f'out_{k}'] = profile_macs(module, args=(h,emb))
            params[f'out_{k}'] = sum(p.numel() for p in module.parameters())
            h = module(h, emb)
        h = h.type(x.dtype)
        profile['out'] = profile_macs(self.out, args=(h,))
        params['out'] = sum(p.numel() for p in self.out.parameters())
        return profile, params


@register_model
def register_diffusion_tts2(opt_net, opt):
    return DiffusionTts(**opt_net['kwargs'])


# Test for ~4 second audio clip at 22050Hz
if __name__ == '__main__':
    clip = torch.randn(4, 1, 86016)
    tok = torch.randint(0,30, (4,388))
    cond = torch.randn(4, 1, 44000)
    ts = torch.LongTensor([555, 556, 600, 600])
    model = DiffusionTts(64, channel_mult=[1,1.5,2, 3, 4, 6, 8, 8, 8, 8], num_res_blocks=[2, 2, 2, 2, 2, 2, 2, 4, 4, 4],
                         token_conditioning_resolutions=[1,4,16,64], attention_resolutions=[256,512], num_heads=4, kernel_size=3,
                         scale_factor=2, conditioning_inputs_provided=True, time_embed_dim_multiplier=4)
    model(clip, ts, tok, cond)

    p, r = model.benchmark(clip, ts, tok, cond)
    p = {k: v / 1000000000 for k, v in p.items()}
    p = sorted(p.items(), key=operator.itemgetter(1))
    print("Computational complexity:")
    print(p)
    print(sum([j[1] for j in p]))
    print()
    print("Memory complexity:")
    r = {k: v / 1000000 for k, v in r.items()}
    r = sorted(r.items(), key=operator.itemgetter(1))
    print(r)
    print(sum([j[1] for j in r]))

