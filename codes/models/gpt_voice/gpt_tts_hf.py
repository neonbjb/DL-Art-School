from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from models.gpt_voice.mini_encoder import AudioMiniEncoder
from models.tacotron2.text import symbols
from trainer.networks import register_model
from utils.util import opt_get


class GptTtsHf(nn.Module):
    NUMBER_TEXT_TOKENS = len(symbols)+1
    START_TEXT_TOKEN = len(symbols)
    STOP_TEXT_TOKEN = 0
    NUMBER_MEL_CODES = 8194
    START_MEL_TOKEN = 8192
    STOP_MEL_TOKEN = 8193

    def __init__(self, layers=8, model_dim=512, heads=8, max_symbols_per_phrase=200, max_mel_tokens=250, max_conditioning_inputs=3, checkpointing=True):
        super().__init__()
        self.max_mel_tokens = max_mel_tokens
        self.max_symbols_per_phrase = max_symbols_per_phrase

        self.model_dim = model_dim
        self.max_mel_tokens = max_mel_tokens
        self.max_conditioning_inputs = max_conditioning_inputs
        self.conditioning_encoder = AudioMiniEncoder(80, model_dim)
        self.text_pos_embedding = nn.Embedding(self.max_symbols_per_phrase + 1, model_dim)
        self.conditioning_embedding = nn.Embedding(self.max_conditioning_inputs, model_dim)
        self.mel_pos_embedding = nn.Embedding(self.max_mel_tokens + 2, model_dim)
        seq_length = 2+self.max_symbols_per_phrase+self.max_conditioning_inputs+self.max_mel_tokens
        self.gpt_config = GPT2Config(vocab_size=self.NUMBER_MEL_CODES,
                                        n_positions=seq_length,
                                        n_ctx=seq_length,
                                        n_embd=model_dim,
                                        n_layer=layers,
                                        n_head=heads,
                                        gradient_checkpointing=checkpointing,
                                        use_cache=not checkpointing)
        self.gpt = GPT2Model(self.gpt_config)
        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.NUMBER_TEXT_TOKENS)
        self.mel_head = nn.Linear(model_dim, self.NUMBER_MEL_CODES)


    def get_logits(self, text_inputs, cond_inputs, mel_targets, get_attns=False):
        assert text_inputs.shape[1] <= self.max_symbols_per_phrase
        assert cond_inputs.shape[1] <= self.max_conditioning_inputs
        assert mel_targets.shape[1] <= self.max_mel_tokens

        mel_targets = F.pad(mel_targets, (1,0), value=self.START_MEL_TOKEN)
        mel_targets = F.pad(mel_targets, (0, self.max_mel_tokens - mel_targets.shape[1]), value=self.STOP_MEL_TOKEN)
        mel_emb = self.gpt.get_input_embeddings()(mel_targets)
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_targets.device))

        text_targets = F.pad(text_inputs, (1,0), value=self.START_TEXT_TOKEN)
        text_targets = F.pad(text_inputs, (0, self.max_symbols_per_phrase - text_targets.shape[1]), value=self.STOP_TEXT_TOKEN)
        text_emb = self.gpt.get_input_embeddings()(text_targets)
        text_emb = text_emb + self.text_pos_embedding(torch.arange(text_emb.shape[1], device=text_targets.device))

        conds = []
        for k in range(cond_inputs.shape[1]):
            conds.append(self.conditioning_encoder(cond_inputs[:, k]))
        while len(conds) < self.max_conditioning_inputs:
            conds.append(conds[-1])
        conds = torch.stack(conds, dim=1)
        conds = conds + self.conditioning_embedding(torch.arange(conds.shape[1], device=conds.device))

        emb = torch.cat([mel_emb, conds, text_emb], dim=1)
        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        if get_attns:
            return gpt_out.attentions
        enc = gpt_out.last_hidden_state

        text_logits = self.final_norm(enc[:, :self.max_symbols_per_phrase])
        text_logits = self.text_head(text_logits)
        text_logits = text_logits.permute(0,2,1)
        mel_logits = self.final_norm(enc[:, -self.max_mel_tokens:])
        mel_logits = self.mel_head(mel_logits)
        mel_logits = mel_logits.permute(0,2,1)

        return text_logits, mel_logits

    def forward(self, text_inputs, cond_inputs, mel_targets, return_attentions=False):
        """
        Forward pass
        text_inputs: long tensor, (b,t)
        cond_inputs: MEL float tensor, (b,c,80,s)
        mel_targets: long tensor, (b,m)
        """
        text_logits, mel_logits = self.get_logits(text_inputs, cond_inputs, mel_targets, get_attns=return_attentions)
        if return_attentions:
            return mel_logits

        text_targets = F.pad(text_inputs, (0,self.max_symbols_per_phrase-text_inputs.shape[1]), value=self.STOP_TEXT_TOKEN)
        loss_text = F.cross_entropy(text_logits, text_targets.long())
        mel_targets = F.pad(mel_targets, (0,self.max_mel_tokens-mel_targets.shape[1]), value=self.STOP_MEL_TOKEN)
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long())
        return loss_text.mean(), loss_mel.mean(), mel_logits


@register_model
def register_gpt_tts_hf(opt_net, opt):
    return GptTtsHf(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    gpt = GptTtsHf()
    l = gpt(torch.randint(high=len(symbols), size=(2,100)),
            torch.randn(2,2,80,800),
            torch.randint(high=8192, size=(2,200)))
