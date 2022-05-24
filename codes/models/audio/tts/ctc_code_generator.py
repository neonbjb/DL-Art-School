from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio.tts.unet_diffusion_tts7 import CheckpointedLayer
from models.lucidrains.x_transformers import Encoder
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
    def __init__(self, model_dim=512, layers=10, max_length=2048, dropout=.1, ctc_codes=256, max_pad=120, max_repeat=30):
        super().__init__()
        self.max_pad = max_pad
        self.max_repeat = max_repeat
        self.ctc_codes = ctc_codes
        pred_codes = (max_pad+1)*(max_repeat+1)

        self.position_embedding = nn.Embedding(max_length, model_dim)
        self.codes_embedding = nn.Embedding(ctc_codes, model_dim)
        self.recursive_embedding = nn.Embedding(pred_codes, model_dim)
        self.mask_embedding = nn.Parameter(torch.randn(model_dim))
        self.encoder = Encoder(
                    dim=model_dim,
                    depth=layers,
                    heads=model_dim//64,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                )
        self.pred_head = nn.Linear(model_dim, pred_codes)
        self.confidence_head = nn.Linear(model_dim, 1)

    def inference(self, codes, pads, repeats):
        position_h = self.position_embedding(torch.arange(0, codes.shape[-1], device=codes.device))
        codes_h = self.codes_embedding(codes)

        labels = pads + repeats * self.max_pad
        mask = labels == 0
        recursive_h = self.recursive_embedding(labels)
        recursive_h[mask] = self.mask_embedding

        h = self.encoder(position_h + codes_h + recursive_h)
        pred_logits = self.pred_head(h)
        confidences = self.confidence_head(h).squeeze(-1)
        confidences = F.softmax(confidences * mask, dim=-1)
        return pred_logits, confidences

    def forward(self, codes, pads, repeats, unpadded_lengths):
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
        assert codes.max() < self.ctc_codes, codes.max()

        labels = pads + repeats * self.max_pad

        position_h = self.position_embedding(torch.arange(0, codes.shape[-1], device=codes.device))
        codes_h = self.codes_embedding(codes)
        recursive_h = self.recursive_embedding(labels)

        mask_prob = random()
        mask = torch.rand_like(labels.float()) > mask_prob
        for b in range(codes.shape[0]):
            mask[b, unpadded_lengths[b]:] = False
        recursive_h[mask.logical_not()] = self.mask_embedding

        h = self.encoder(position_h + codes_h + recursive_h)
        pred_logits = self.pred_head(h)
        loss = F.cross_entropy(pred_logits.permute(0,2,1), labels, reduce=False)

        confidences = self.confidence_head(h).squeeze(-1)
        confidences = F.softmax(confidences * mask, dim=-1)
        confidence_loss = loss * confidences
        loss = loss / loss.shape[-1]  # This balances the confidence_loss and loss.

        return loss.mean(), confidence_loss.mean()


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