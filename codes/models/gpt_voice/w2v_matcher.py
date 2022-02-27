import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder, Decoder, ContinuousTransformerWrapper

from models.gpt_voice.mini_encoder import AudioMiniEncoder


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


class CheckpointedXTransformer(nn.Module):
    """
    Wraps a ContinuousTransformerWrapper and applies CheckpointedLayer to each layer and permutes from channels-mid
    to channels-last that XTransformer expects.
    """
    def __init__(self, **xtransformer_kwargs):
        super().__init__()
        self.transformer = ContinuousTransformerWrapper(**xtransformer_kwargs)

        for i in range(len(self.transformer.attn_layers.layers)):
            n, b, r = self.transformer.attn_layers.layers[i]
            self.transformer.attn_layers.layers[i] = nn.ModuleList([n, CheckpointedLayer(b), r])

    def forward(self, x, **kwargs):
        return self.transformer(x, **kwargs)


class Wav2VecMatcher(nn.Module):
    def __init__(self,
                 model_dim,
                 encoder_depth,
                 decoder_depth,
                 num_text_tokens=148,
                 dropout=.1):
        super().__init__()

        WAV2VEC_CHANNELS = 1024
        self.conditioning_encoder = AudioMiniEncoder(1, model_dim, base_channels=32, depth=6, resnet_blocks=1,
                         attn_blocks=2, num_attn_heads=2, dropout=dropout, downsample_factor=4, kernel_size=5)
        self.text_embedding = nn.Embedding(num_text_tokens, model_dim)
        self.encoder = CheckpointedXTransformer(
                max_seq_len=-1,
                use_pos_emb=False,
                attn_layers=Encoder(
                    dim=model_dim,
                    depth=encoder_depth,
                    heads=model_dim//64,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_emb_dim=True,
                )
            )
        self.decoder_start_embedding = nn.Parameter(torch.randn(1,1,model_dim))
        self.decoder_stop_embedding = nn.Parameter(torch.randn(1,model_dim))
        self.w2v_query_encoder = nn.Linear(WAV2VEC_CHANNELS, model_dim)
        self.w2v_value_encoder = nn.Linear(WAV2VEC_CHANNELS, model_dim)
        self.decoder = CheckpointedXTransformer(
                max_seq_len=-1,  # Should be unused
                use_pos_emb=False,
                attn_layers=Decoder(
                    dim=model_dim,
                    depth=decoder_depth,
                    heads=model_dim//64,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                    cross_attend=True,
                )
        )

    def forward(self, text_tokens, conditioning_clip, w2v_logits, token_lengths, w2v_lengths):
        # Clip off text_lengths where possible to save compute.
        max_text_len = token_lengths.max()
        text_tokens = text_tokens[:, :max_text_len]

        text_emb = self.text_embedding(text_tokens)
        cond_emb = self.conditioning_encoder(conditioning_clip)
        enc_inputs = torch.cat([cond_emb.unsqueeze(1), text_emb], dim=1)
        dec_context = self.encoder(enc_inputs)
        w2v_values = self.w2v_value_encoder(w2v_logits)
        dec_inputs = torch.cat([self.decoder_start_embedding.repeat(w2v_values.shape[0],1,1), w2v_values], dim=1)
        dec_out = self.decoder(dec_inputs, context=dec_context)[:, :-1]
        w2v_queries = self.w2v_query_encoder(w2v_logits)

        # Compute loss
        b,l,c = dec_out.shape
        keys_uncompressed = dec_out.reshape(b*l, c)
        queries_uncompressed = w2v_queries.reshape(b*l, c)
        dot = torch.einsum("i c, j c -> i j", keys_uncompressed, queries_uncompressed)
        labels = torch.arange(0, b*l, 1, device=dot.device)
        # TODO: weight the cross entropy: logits from the same clip should be weighted as possible "matches" (say, share ~10% of the probability mass). Logits near
        #       the w2v logits should also get a bump in probability mass. Cross entropy is probably not the right avenue for this. This is important to enable
        #       "searching" for w2v matches from a large pool.
        ce_loss1 = F.cross_entropy(dot, labels, reduction="none")
        ce_loss2 = F.cross_entropy(dot.t(), labels, reduction="none")
        mse_pad_loss = F.mse_loss(keys_uncompressed, self.decoder_stop_embedding.repeat(b*l,1), reduction="none").sum(dim=-1)

        # Create a mask based on w2v_lengths that will be used to ensure the encodings of padding tokens are not considered in the cross entropy loss
        loss_mask = torch.ones((b,l), device=ce_loss1.device)
        for i in range(b):
            loss_mask[i, w2v_lengths[i]:] = 0
        loss_mask = loss_mask.reshape(b*l)

        ce_loss = (ce_loss1 * loss_mask + ce_loss2 * loss_mask).mean()
        mse_loss = (mse_pad_loss * (loss_mask == 0)).mean()

        return ce_loss, mse_loss


if __name__ == '__main__':
    model = Wav2VecMatcher(512, 8, 8)
    toks = torch.randint(0, 100, (4,100))
    tok_lens = torch.tensor([50,60,70,80])
    cond = torch.randn(4,1,44000)
    logits = torch.randn(4,120,1024)
    logit_lens = torch.tensor([60,70,80,90])
    model(toks, cond, logits, tok_lens, logit_lens)