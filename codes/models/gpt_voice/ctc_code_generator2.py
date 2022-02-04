import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5Model
from x_transformers import Encoder, XTransformer

from models.gpt_voice.transformer_builders import null_position_embeddings
from models.gpt_voice.unet_diffusion_tts6 import CheckpointedLayer
from models.gpt_voice.unified_voice2 import ConditioningEncoder
from trainer.networks import register_model
from utils.util import opt_get


class CtcCodeGenerator(nn.Module):
    def __init__(self, model_dim=512, layers=10, num_heads=8, dropout=.1, ctc_codes=36, max_pad=120, max_repeat=30, checkpointing=True):
        super().__init__()
        self.max_pad = max_pad
        self.max_repeat = max_repeat
        self.start_token = (self.max_repeat+1)*(self.max_pad+1)+1
        self.conditioning_encoder = ConditioningEncoder(80, model_dim, num_attn_heads=num_heads)
        self.embedding = nn.Embedding(ctc_codes, model_dim)
        self.dec_embedding = nn.Embedding(self.start_token+1, model_dim)
        self.config = T5Config(
            vocab_size=1,  # T5 embedding will be removed and replaced with custom embedding.
            d_model=model_dim,
            d_kv=model_dim//num_heads,
            d_ff=model_dim*4,
            num_layers=layers,
            num_heads=num_heads,
            dropout_rate=dropout,
            feed_forward_proj='gated-gelu',
            use_cache=not checkpointing,
            gradient_checkpointing=checkpointing
        )
        self.transformer = T5Model(self.config)
        del self.transformer.encoder.embed_tokens
        del self.transformer.decoder.embed_tokens
        self.transformer.encoder.embed_tokens = functools.partial(null_position_embeddings, dim=model_dim)
        self.transformer.decoder.embed_tokens = functools.partial(null_position_embeddings, dim=model_dim)
        self.output_layer = nn.Linear(model_dim, self.start_token+1)


    def forward(self, conditioning_input, codes, pads, repeats, unpadded_lengths):
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

        conditioning_input = conditioning_input.unsqueeze(1) if len(conditioning_input.shape) == 3 else conditioning_input
        conds = []
        for j in range(conditioning_input.shape[1]):
            conds.append(self.conditioning_encoder(conditioning_input[:, j]))
        conds = torch.stack(conds, dim=1)
        h = torch.cat([conds, self.embedding(codes)], dim=1)

        labels = pads + repeats * self.max_pad + 1
        for i in range(unpadded_lengths.shape[0]):
            labels[i, unpadded_lengths[i]:] = 0
        labels_in = F.pad(labels, (1,0), value=self.start_token)
        h_dec = self.dec_embedding(labels_in)

        h = self.transformer(inputs_embeds=h, decoder_inputs_embeds=h_dec).last_hidden_state
        logits = self.output_layer(h)
        logits = logits.permute(0,2,1)[:,:,:-1]  # Strip off the last token. There is no "stop" token here, so this is just an irrelevant prediction on some future that doesn't actually exist.
        loss = F.cross_entropy(logits, labels, reduction='none')

        # Ignore the first predictions of the sequences. This corresponds to the padding for the first CTC character, which is pretty much random and cannot be predicted.
        #loss = loss[1:].mean()
        return loss


@register_model
def register_ctc_code_generator2(opt_net, opt):
    return CtcCodeGenerator(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    model = CtcCodeGenerator()
    conds = torch.randn(4,2,80,600)
    inps = torch.randint(0,36, (4, 300))
    pads = torch.randint(0,100, (4,300))
    repeats = torch.randint(0,20, (4,300))
    loss = model(conds, inps, pads, repeats, torch.tensor([250, 300, 280, 30]))
    print(loss.shape)