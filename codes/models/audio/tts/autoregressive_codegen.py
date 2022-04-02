import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import XTransformer, TransformerWrapper, Encoder, Decoder

from models.arch_util import AttentionBlock
from trainer.networks import register_model


class ResBlock(nn.Module):
    """
    Basic residual convolutional block that uses GroupNorm.
    """
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan//8, chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan//8, chan)
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False):
        super().__init__()
        attn = []
        self.init = nn.Sequential(nn.Conv1d(spec_dim, embedding_dim//4, kernel_size=5, padding=2),
                                  nn.Conv1d(embedding_dim//4, embedding_dim//2, kernel_size=3, padding=1, stride=2),
                                  ResBlock(embedding_dim//2),
                                  nn.Conv1d(embedding_dim//2, embedding_dim, kernel_size=3, padding=1, stride=2))
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads, do_checkpoint=do_checkpointing))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        return h.mean(dim=2)


class CheckpointedLayer(nn.Module):
    """
    Wraps a module. When forward() is called, passes kwargs that require_grad through torch.checkpoint() and bypasses
    checkpoint for all other args.
    """
    def __init__(self, wrap):
        super().__init__()
        self.wrap = wrap

    def forward(self, x, *args, **kwargs):
        for k, v in kwargs.items():
            assert not (isinstance(v, torch.Tensor) and v.requires_grad)  # This would screw up checkpointing.
        partial = functools.partial(self.wrap, **kwargs)
        return torch.utils.checkpoint.checkpoint(partial, x, *args)


class CheckpointedXTransformerWrapper(nn.Module):
    """
    Wraps a TransformerWrapper and applies CheckpointedLayer to each layer.
    """
    def __init__(self, checkpoint=True, **xtransformer_kwargs):
        super().__init__()
        self.transformer = TransformerWrapper(**xtransformer_kwargs)

        if not checkpoint:
            return
        for i in range(len(self.transformer.attn_layers.layers)):
            n, b, r = self.transformer.attn_layers.layers[i]
            self.transformer.attn_layers.layers[i] = nn.ModuleList([n, CheckpointedLayer(b), r])

    def forward(self, x, **kwargs):
        return self.transformer(x, **kwargs)


class AutoregressiveCodegen(nn.Module):
    def __init__(self, model_dim, depth, num_text_tokens=256, num_mel_tokens=8194, max_text_tokens=4000,
                 max_mel_tokens=4000, dropout=.1):
        super().__init__()

        self.START_TOKEN=8192
        self.STOP_TOKEN=8193
        self.mel_embedding = ConditioningEncoder(80, model_dim, do_checkpointing=False)
        self.encoder = CheckpointedXTransformerWrapper(
                                  num_tokens=num_text_tokens,
                                  max_seq_len=max_text_tokens,
                                  attn_layers = Encoder(
                                      depth=depth//2,
                                      heads=model_dim//64,
                                      dim=model_dim,
                                      attn_dropout=dropout,
                                      ff_dropout=dropout,
                                      use_rmsnorm=True,
                                      ff_glu=True,
                                      ff_mult=1,
                                      rotary_pos_emb=True,
                                      rel_pos_bias=True,
                                  ))
        self.decoder = CheckpointedXTransformerWrapper(
                                  num_tokens=num_mel_tokens,
                                  max_seq_len=max_mel_tokens,
                                  attn_layers=Decoder(
                                      depth=depth,
                                      heads=model_dim//64,
                                      dim=model_dim,
                                      attn_dropout=dropout,
                                      ff_dropout=dropout,
                                      use_rmsnorm=True,
                                      ff_glu=True,
                                      ff_mult=1,
                                      rotary_pos_emb=True,
                                      rel_pos_bias=True,
                                      cross_attend=True,
                                  ))

    def get_grad_norm_parameter_groups(self):
        return {
            'encoder': list(self.encoder.parameters()),
            'decoder': list(self.decoder.parameters()),
            'minicoder': list(self.mel_embedding.parameters()),
        }

    def forward(self, text_codes, conditioning_signal, mel_codes, wav_lengths, return_loss=True):
        # Format mel_codes with a stop token on the end.
        mel_lengths = wav_lengths // 1024 + 1
        for b in range(mel_codes.shape[0]):
            mel_codes[b, mel_lengths[b]:] = self.STOP_TOKEN
        mel_codes = F.pad(mel_codes, (0, 1), value=self.STOP_TOKEN)

        if len(conditioning_signal.shape) != 4:
            conditioning_signal = conditioning_signal.unsqueeze(1)
        cond_embs = []
        for i in range(conditioning_signal.shape[1]):
            cond_embs.append(self.mel_embedding(conditioning_signal[:, i]))
        cond_emb = torch.stack(cond_embs, dim=1).mean(dim=1, keepdim=True)
        enc_text = self.encoder(text_codes, return_embeddings=True)
        context = torch.cat([cond_emb, enc_text], dim=1)
        dec_inputs = F.pad(mel_codes, (1,0), value=self.START_TOKEN)[:, :-1]
        dec = self.decoder(dec_inputs, context=context)
        if not return_loss:
            return dec
        loss_mel = F.cross_entropy(dec.permute(0,2,1), mel_codes)
        return loss_mel


@register_model
def register_autoregressive_codegen(opt_net, opt):
    return AutoregressiveCodegen(**opt_net['kwargs'])


if __name__ == '__main__':
    codegen = AutoregressiveCodegen(1024, 20)
    torch.save(codegen.state_dict(), 'sample.pth')
    codegen(torch.randint(0,256, (2,200)),
            torch.randn(2,80,120),
            torch.randint(0,8192, (2,350)),
            torch.tensor([192,350]))