import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import AttentionBlock, TimestepEmbedSequential, \
    Downsample, Upsample, TimestepBlock
from models.lucidrains.x_transformers import Encoder
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
    def __init__(
            self,
            model_channels,
            in_channels=100,
            num_tokens=256,
            out_channels=200,  # mean and variance
            dropout=0,
            # m                 1,  2,   4,   8
            block_channels=  (512,640, 768,1024),
            num_res_blocks=  (3,    3,   3,   3),
            token_conditioning_resolutions=(2,4,8),
            attention_resolutions=(2,4,8),
            conv_resample=True,
            dims=1,
            use_fp16=False,
            kernel_size=3,
            scale_factor=2,
            time_embed_dim_multiplier=4,
            nil_guidance_fwd_proportion=.15,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.conv_resample = conv_resample
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.dims = dims
        self.nil_guidance_fwd_proportion = nil_guidance_fwd_proportion
        self.mask_token_id = num_tokens
        num_heads = model_channels // 64

        padding = 1 if kernel_size == 3 else 2

        time_embed_dim = model_channels * time_embed_dim_multiplier
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.code_embedding = nn.Embedding(num_tokens+1, model_channels)
        self.conditioning_embedder = nn.Sequential(nn.Conv1d(in_channels, model_channels // 2, 3, padding=1, stride=2),
                                                   nn.Conv1d(model_channels//2, model_channels,3,padding=1,stride=2))
        self.conditioning_encoder = Encoder(
                    dim=model_channels,
                    depth=4,
                    heads=num_heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                )

        self.codes_encoder = Encoder(
                    dim=model_channels,
                    depth=8,
                    heads=num_heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rms_scaleshift_norm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                    zero_init_branch_output=True,
                )

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

        for level, (blk_chan, num_blocks) in enumerate(zip(block_channels, num_res_blocks)):
            if ds in token_conditioning_resolutions:
                token_conditioning_block = nn.Conv1d(model_channels, ch, 1)
                token_conditioning_block.weight.data *= .02
                self.input_blocks.append(token_conditioning_block)
                token_conditioning_blocks.append(token_conditioning_block)

            for _ in range(num_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=blk_chan,
                        dims=dims,
                        kernel_size=kernel_size,
                    )
                ]
                ch = blk_chan
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(block_channels) - 1:
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
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, (blk_chan, num_blocks) in list(enumerate(zip(block_channels, num_res_blocks)))[::-1]:
            for i in range(num_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=blk_chan,
                        dims=dims,
                        kernel_size=kernel_size,
                    )
                ]
                ch = blk_chan
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
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
        with autocast(x.device.type):
            orig_x_shape = x.shape[-1]
            cm = ceil_multiple(x.shape[-1], 16)
            if cm != 0:
                pc = (cm-x.shape[-1])/x.shape[-1]
                x = F.pad(x, (0,cm-x.shape[-1]))
                tokens = F.pad(tokens, (0,int(pc*tokens.shape[-1])))

            hs = []
            time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

            # Mask out guidance tokens for un-guided diffusion.
            if self.training and self.nil_guidance_fwd_proportion > 0:
                token_mask = torch.rand(tokens.shape, device=tokens.device) < self.nil_guidance_fwd_proportion
                tokens = torch.where(token_mask, self.mask_token_id, tokens)
            code_emb = self.code_embedding(tokens).permute(0,2,1)
            cond_emb = self.conditioning_embedder(conditioning_input).permute(0,2,1)
            cond_emb = self.conditioning_encoder(cond_emb)[:, 0]
            code_emb = self.codes_encoder(code_emb.permute(0,2,1), norm_scale_shift_inp=cond_emb).permute(0,2,1)

            first = True
            time_emb = time_emb.float()
            h = x
            for k, module in enumerate(self.input_blocks):
                if isinstance(module, nn.Conv1d):
                    h_tok = F.interpolate(module(code_emb), size=(h.shape[-1]), mode='nearest')
                    h = h + h_tok
                else:
                    with autocast(x.device.type, enabled=not first):
                        # First block has autocast disabled to allow a high precision signal to be properly vectorized.
                        h = module(h, time_emb)
                    hs.append(h)
                first = False
            h = self.middle_block(h, time_emb)
            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, time_emb)

        # Last block also has autocast disabled for high-precision outputs.
        h = h.float()
        out = self.out(h)
        return out[:, :, :orig_x_shape]


@register_model
def register_diffusion_tts10(opt_net, opt):
    return DiffusionTts(**opt_net['kwargs'])


if __name__ == '__main__':
    clip = torch.randn(2, 100, 500).cuda()
    tok = torch.randint(0,256, (2,230)).cuda()
    cond = torch.randn(2, 100, 300).cuda()
    ts = torch.LongTensor([600, 600]).cuda()
    model = DiffusionTts(512).cuda()
    print(sum(p.numel() for p in model.parameters()) / 1000000)
    model(clip, ts, tok, cond)

