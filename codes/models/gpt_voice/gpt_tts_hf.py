from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from models.gpt_voice.gpt_asr_hf import GPT2InferenceModel
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

    def __init__(self, layers=8, model_dim=512, heads=8, max_symbols_per_phrase=200, max_mel_tokens=250, max_conditioning_inputs=3,
                 checkpointing=True, mel_length_compression=1024):
        super().__init__()
        self.max_mel_tokens = max_mel_tokens
        self.max_symbols_per_phrase = max_symbols_per_phrase
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.conditioning_encoder = AudioMiniEncoder(80, model_dim)
        self.text_embedding = nn.Embedding(self.NUMBER_TEXT_TOKENS, model_dim)
        self.text_pos_embedding = nn.Embedding(self.max_symbols_per_phrase + 1, model_dim)
        self.conditioning_embedding = nn.Parameter(torch.randn(1,model_dim), requires_grad=True)
        self.mel_pos_embedding = nn.Embedding(self.max_mel_tokens + 1, model_dim)
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


    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1,0), value=start_token)
        tar = F.pad(input, (0,1), value=stop_token)
        return inp, tar

    def get_logits(self, text_inputs, cond_inputs, mel_inputs, get_attns=False):
        text_emb = self.text_embedding(text_inputs)
        text_emb = text_emb + self.text_pos_embedding(torch.arange(text_emb.shape[1], device=text_inputs.device))

        conds = []
        for k in range(cond_inputs.shape[1]):
            conds.append(self.conditioning_encoder(cond_inputs[:, k]))
        while len(conds) < self.max_conditioning_inputs:
            conds.append(conds[-1])
        conds = torch.stack(conds, dim=1)
        conds = conds + self.conditioning_embedding

        mel_emb = self.gpt.get_input_embeddings()(mel_inputs)
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_inputs.device))

        emb = torch.cat([text_emb, conds, mel_emb], dim=1)
        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        if get_attns:
            return gpt_out.attentions
        enc = gpt_out.last_hidden_state

        text_logits = self.final_norm(enc[:, :text_emb.shape[1]])
        text_logits = self.text_head(text_logits)
        text_logits = text_logits.permute(0,2,1)
        mel_logits = self.final_norm(enc[:, -mel_emb.shape[1]:])
        mel_logits = self.mel_head(mel_logits)
        mel_logits = mel_logits.permute(0,2,1)

        return text_logits, mel_logits

    def forward(self, text_inputs, cond_inputs, mel_targets, wav_lengths, return_attentions=False):
        """
        Forward pass
        text_inputs: long tensor, (b,t)
        cond_inputs: MEL float tensor, (b,c,80,s)
        mel_targets: long tensor, (b,m)
        mel_lengths: long tensor, (b,)
        """
        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>)
        mel_lengths = wav_lengths // self.mel_length_compression
        for b in range(len(mel_lengths)):
            if mel_lengths[b] < mel_targets.shape[-1]:
                mel_targets[b, mel_lengths[b]:] = self.STOP_MEL_TOKEN

        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.START_TEXT_TOKEN, self.STOP_TEXT_TOKEN)
        mel_inputs, mel_targets = self.build_aligned_inputs_and_targets(mel_targets, self.START_MEL_TOKEN, self.STOP_MEL_TOKEN)
        text_logits, mel_logits = self.get_logits(text_inputs, cond_inputs, mel_inputs, get_attns=return_attentions)
        if return_attentions:
            return mel_logits
        loss_text = F.cross_entropy(text_logits, text_targets.long())
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long())
        return loss_text.mean(), loss_mel.mean(), mel_logits

    def inference(self, text_inputs, cond_inputs, do_sample=False, temperature=1.0, num_beams=8, repetition_penalty=1):
        #text_inputs, cond_inputs = torch.load("debug_text_and_cond.pt")

        if not hasattr(self, 'inference_model'):
            self.inference_model = GPT2InferenceModel(self.gpt_config, self.gpt, self.mel_pos_embedding, self.final_norm, self.mel_head)

        text_inputs = F.pad(text_inputs, (0, self.max_symbols_per_phrase - text_inputs.shape[1]), value=self.STOP_TEXT_TOKEN)
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.START_TEXT_TOKEN, self.STOP_TEXT_TOKEN)
        text_emb = self.text_embedding(text_inputs)
        text_emb = text_emb + self.text_pos_embedding(torch.arange(text_emb.shape[1], device=text_inputs.device))

        conds = []
        for k in range(cond_inputs.shape[1]):
            conds.append(self.conditioning_encoder(cond_inputs[:, k]))
        while len(conds) < self.max_conditioning_inputs:
            conds.append(conds[-1])
        conds = torch.stack(conds, dim=1)
        conds = conds + self.conditioning_embedding

        emb = torch.cat([text_emb, conds], dim=1)
        self.inference_model.store_mel_emb(emb)

        fake_inputs = torch.full((emb.shape[0],emb.shape[1]+1,), fill_value=1, dtype=torch.long, device=text_inputs.device)
        fake_inputs[:,-1] = self.START_MEL_TOKEN

        gen = self.inference_model.generate(fake_inputs, do_sample=do_sample, bos_token_id=self.START_MEL_TOKEN, pad_token_id=self.STOP_MEL_TOKEN, eos_token_id=self.STOP_MEL_TOKEN,
                          max_length=emb.shape[1]+self.max_mel_tokens, temperature=temperature, num_beams=num_beams, use_cache=True, repetition_penalty=repetition_penalty)
        return gen[:, fake_inputs.shape[1]:-1]


@register_model
def register_gpt_tts_hf(opt_net, opt):
    return GptTtsHf(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    gpt = GptTtsHf(model_dim=1024, heads=16)
    l = gpt(torch.randint(high=len(symbols), size=(2,200)),
            torch.randn(2,2,80,800),
            torch.randint(high=8192, size=(2,250)),
            torch.tensor([150*256,195*256]))
