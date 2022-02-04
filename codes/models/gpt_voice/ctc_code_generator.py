import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder, XTransformer

from models.gpt_voice.unet_diffusion_tts6 import CheckpointedLayer
from trainer.networks import register_model
from utils.util import opt_get


class CheckpointedXTransformerEncoder(nn.Module):
    """
    Wraps a ContinuousTransformerWrapper and applies CheckpointedLayer to each layer and permutes from channels-mid
    to channels-last that XTransformer expects.
    """
    def __init__(self, **xtransformer_kwargs):
        super().__init__()
        self.transformer = XTransformer(**xtransformer_kwargs)

        for xform in [self.transformer.encoder, self.transformer.decoder.net]:
            for i in range(len(xform.attn_layers.layers)):
                n, b, r = xform.attn_layers.layers[i]
                xform.attn_layers.layers[i] = nn.ModuleList([n, CheckpointedLayer(b), r])

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)


class CtcCodeGenerator(nn.Module):
    def __init__(self, model_dim=512, layers=10, num_heads=8, dropout=.1, ctc_codes=36, max_pad=120, max_repeat=30):
        super().__init__()
        self.max_pad = max_pad
        self.max_repeat = max_repeat
        self.transformer = XTransformer(
            dim=model_dim,
            enc_depth=layers,
            dec_depth=layers,
            enc_heads=num_heads,
            dec_heads=num_heads,
            enc_num_tokens=ctc_codes,
            dec_num_tokens=(max_pad+1)*(max_repeat+1),
            enc_max_seq_len=-1,
            dec_max_seq_len=-1,

            enc_ff_dropout=dropout,
            enc_attn_dropout=dropout,
            enc_use_rmsnorm=True,
            enc_ff_glu=True,
            enc_rotary_pos_emb=True,
            dec_ff_dropout=dropout,
            dec_attn_dropout=dropout,
            dec_use_rmsnorm=True,
            dec_ff_glu=True,
            dec_rotary_pos_emb=True)

    def forward(self, codes, pads, repeats, unpadded_lengths=None):
        if unpadded_lengths is not None:
            max_len = unpadded_lengths.max()
            codes = codes[:, :max_len]
            pads = pads[:, :max_len]
            repeats = repeats[:, :max_len]

        if pads.max() > self.max_pad:
            print(f"Got unexpectedly long pads. Max: {pads.max()}, {pads}")
            pads = torch.clip(pads, 0, self.max_pad)
        if repeats.max() > self.max_repeat:
            print(f"Got unexpectedly long repeats. Max: {repeats.max()}, {repeats}")
            repeats = torch.clip(repeats, 0, self.max_repeat)
        assert codes.max() < 36, codes.max()

        labels = pads + repeats * self.max_pad
        loss = self.transformer(codes, labels)
        return loss


@register_model
def register_ctc_code_generator(opt_net, opt):
    return CtcCodeGenerator(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    model = CtcCodeGenerator()
    inps = torch.randint(0,36, (4, 300))
    pads = torch.randint(0,100, (4,300))
    repeats = torch.randint(0,20, (4,300))
    loss = model(inps, pads, repeats)
    print(loss.shape)