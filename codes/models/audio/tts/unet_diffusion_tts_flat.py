import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder

from models.audio.tts.diffusion_encoder import TimestepEmbeddingAttentionLayers
from models.audio.tts.mini_encoder import AudioMiniEncoder
from models.audio.tts.unet_diffusion_tts7 import CheckpointedXTransformerEncoder
from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from trainer.networks import register_model


def is_latent(t):
    return t.dtype == torch.float

def is_sequence(t):
    return t.dtype == torch.long


class DiffusionTtsFlat(nn.Module):
    def __init__(
            self,
            model_channels=512,
            num_layers=8,
            in_channels=100,
            in_latent_channels=512,
            in_tokens=8193,
            max_timesteps=4000,
            out_channels=200,  # mean and variance
            dropout=0,
            use_fp16=False,
            num_heads=16,
            # Parameters for regularization.
            layer_drop=.1,
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_heads = num_heads
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.layer_drop = layer_drop

        self.inp_block = nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1)
        time_embed_dim = model_channels
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Either code_converter or latent_converter is used, depending on what type of conditioning data is fed.
        # This model is meant to be able to be trained on both for efficiency purposes - it is far less computationally
        # complex to generate tokens, while generating latents will normally mean propagating through a deep autoregressive
        # transformer network.
        self.code_converter = nn.Sequential(
            nn.Embedding(in_tokens, model_channels),
            CheckpointedXTransformerEncoder(
                needs_permute=False,
                max_seq_len=-1,
                use_pos_emb=False,
                attn_layers=Encoder(
                    dim=model_channels,
                    depth=3,
                    heads=num_heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_emb_dim=True,
                )
            )
        )
        self.latent_converter = nn.Conv1d(in_latent_channels, model_channels, 1)
        if in_channels > 60:  # It's a spectrogram.
            self.contextual_embedder = nn.Sequential(nn.Conv1d(in_channels,model_channels,3,padding=1,stride=2),
                                                     CheckpointedXTransformerEncoder(
                                                         needs_permute=True,
                                                         max_seq_len=-1,
                                                         use_pos_emb=False,
                                                         attn_layers=Encoder(
                                                             dim=model_channels,
                                                             depth=4,
                                                             heads=num_heads,
                                                             ff_dropout=dropout,
                                                             attn_dropout=dropout,
                                                             use_rmsnorm=True,
                                                             ff_glu=True,
                                                             rotary_emb_dim=True,
                                                         )
                                                     ))
        else:
            self.contextual_embedder = AudioMiniEncoder(1, model_channels, base_channels=32, depth=6, resnet_blocks=1,
                                                        attn_blocks=3, num_attn_heads=8, dropout=dropout, downsample_factor=4, kernel_size=5)
        self.conditioning_conv = nn.Conv1d(model_channels*2, model_channels, 1)
        self.unconditioned_embedding = nn.Parameter(torch.randn(1,model_channels,1))
        self.conditioning_timestep_integrator = CheckpointedXTransformerEncoder(
                needs_permute=True,
                max_seq_len=-1,
                use_pos_emb=False,
                attn_layers=TimestepEmbeddingAttentionLayers(
                    dim=model_channels,
                    timestep_dim=time_embed_dim,
                    depth=3,
                    heads=num_heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_emb_dim=True,
                    layerdrop_percent=0,
                )
            )
        self.integrate_conditioning = nn.Conv1d(model_channels*2, model_channels, 1)

        self.layers = CheckpointedXTransformerEncoder(
                needs_permute=True,
                max_seq_len=-1,
                use_pos_emb=False,
                attn_layers=TimestepEmbeddingAttentionLayers(
                    dim=model_channels,
                    timestep_dim=time_embed_dim,
                    depth=num_layers,
                    heads=num_heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_emb_dim=True,
                    layerdrop_percent=layer_drop,
                    zero_init_branch_output=True,
                )
            )
        self.layers.transformer.norm = nn.Identity()  # We don't want the final norm for the main encoder.

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(conv_nd(1, model_channels, out_channels, 3, padding=1)),
        )

    def get_grad_norm_parameter_groups(self):
        groups = {
            'minicoder': list(self.contextual_embedder.parameters()),
            'conditioning_timestep_integrator': list(self.conditioning_timestep_integrator.parameters()),
            'layers': list(self.layers.parameters()),
        }
        return groups

    def forward(self, x, timesteps, aligned_conditioning, conditioning_input, conditioning_free=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param aligned_conditioning: an aligned latent or sequence of tokens providing useful data about the sample to be produced.
        :param conditioning_input: a full-resolution audio clip that is used as a reference to the style you want decoded.
        :param lr_input: for super-sampling models, a guidance audio clip at a lower sampling rate.
        :param conditioning_free: When set, all conditioning inputs (including tokens and conditioning_input) will not be considered.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # Shuffle aligned_latent to BxCxS format
        if is_latent(aligned_conditioning):
            aligned_conditioning = aligned_conditioning.permute(0, 2, 1)

        # Note: this block does not need to repeated on inference, since it is not timestep-dependent or x-dependent.
        unused_params = []
        if conditioning_free:
            code_emb = self.unconditioned_embedding.repeat(x.shape[0], 1, 1)
        else:
            unused_params.append(self.unconditioned_embedding)
            cond_emb = self.contextual_embedder(conditioning_input)
            if len(cond_emb.shape) == 3:  # Just take the first element.
                cond_emb = cond_emb[:, :, 0]
            if is_latent(aligned_conditioning):
                code_emb = self.latent_converter(aligned_conditioning)
                unused_params.extend(list(self.code_converter.parameters()))
            else:
                code_emb = self.code_converter(aligned_conditioning)
                unused_params.extend(list(self.latent_converter.parameters()))
            cond_emb_spread = cond_emb.unsqueeze(-1).repeat(1, 1, code_emb.shape[-1])
            code_emb = self.conditioning_conv(torch.cat([cond_emb_spread, code_emb], dim=1))
        # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
        if self.training and self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand((code_emb.shape[0], 1, 1),
                                               device=code_emb.device) < self.unconditioned_percentage
            code_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(x.shape[0], 1, 1),
                                   code_emb)

        # Everything after this comment is timestep dependent.
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        code_emb = self.conditioning_timestep_integrator(code_emb, time_emb=time_emb)
        x = self.inp_block(x)
        x = self.integrate_conditioning(torch.cat([x, F.interpolate(code_emb, size=x.shape[-1], mode='nearest')], dim=1))
        with torch.autocast(x.device.type, enabled=self.enable_fp16):
            x = self.layers(x, time_emb=time_emb)
        x = x.float()
        out = self.out(x)

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        out = out + extraneous_addition * 0

        return out


@register_model
def register_diffusion_tts_flat(opt_net, opt):
    return DiffusionTtsFlat(**opt_net['kwargs'])


if __name__ == '__main__':
    clip = torch.randn(2, 100, 400)
    aligned_latent = torch.randn(2,388,512)
    aligned_sequence = torch.randint(0,8192,(2,388))
    cond = torch.randn(2, 100, 400)
    ts = torch.LongTensor([600, 600])
    model = DiffusionTtsFlat(512, layer_drop=.3)
    # Test with latent aligned conditioning
    o = model(clip, ts, aligned_latent, cond)
    # Test with sequence aligned conditioning
    o = model(clip, ts, aligned_sequence, cond)

