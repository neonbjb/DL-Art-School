import random
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from models.arch_util import AttentionBlock
from models.gpt_voice.gpt_asr_hf import GPT2InferenceModel
from models.gpt_voice.mini_encoder import AudioMiniEncoder
from models.tacotron2.text import symbols
from trainer.networks import register_model
from utils.util import opt_get


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads, do_checkpoint=do_checkpointing))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        return h[:, :, 0]


class UnifiedGptVoice(nn.Module):
    """
    Derived from GptTtsHf, but offers multiple modes of autoregressive operation:
    - Text only
    - Voice only
    - Text conditioned on voice
    - Voice conditioned on text
    """

    NUMBER_TEXT_TOKENS = 10000  # The number of tokens produced by our bespoke BPE tokenizer.
    START_TEXT_TOKEN = 9999
    STOP_TEXT_TOKEN = 0
    NUMBER_MEL_CODES = 8194
    START_MEL_TOKEN = 8192
    STOP_MEL_TOKEN = 8193

    def __init__(self, layers=8, model_dim=512, heads=8, max_symbols_per_phrase=80, max_mel_tokens=250, max_conditioning_inputs=3,
                 checkpointing=True, mel_length_compression=1024, max_conditioning_length=60):
        super().__init__()


        self.max_mel_tokens = max_mel_tokens
        self.max_symbols_per_phrase = max_symbols_per_phrase
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.conditioning_encoder = ConditioningEncoder(80, model_dim, num_attn_heads=heads)
        self.text_embedding = nn.Embedding(self.NUMBER_TEXT_TOKENS, model_dim)
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
        self.max_conditioning_length = max_conditioning_length


    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1,0), value=start_token)
        tar = F.pad(input, (0,1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, wav_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        mel_lengths = wav_lengths // self.mel_length_compression
        for b in range(len(mel_lengths)):
            actual_end = mel_lengths[b] + 1  # Due to the convolutional nature of how these tokens are generated, it would be best if the model predicts a token past the actual last token.
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.STOP_MEL_TOKEN
        return mel_input_tokens

    def randomly_permute_conditioning_input(self, speech_conditioning_input):
        """
        Randomly permute the conditioning spectrogram, to destroy any structure present. Note that since the
        conditioning input is derived from a discrete spectrogram, it does actually retain structure, but only a little
        bit (actually: exactly how much we want; enough to discriminate different vocal qualities, but nothing about
        what is being said).
        """
        cond_input = speech_conditioning_input[:,:,torch.randperm(speech_conditioning_input.shape[-1])]
        if cond_input.shape[-1] > self.max_conditioning_length:
            cond_input = cond_input[:,:,:self.max_conditioning_length]
        return cond_input

    def get_logits(self, speech_conditioning_input, first_inputs, first_head, second_inputs=None, second_head=None, get_attns=False):
        if second_inputs is not None:
            emb = torch.cat([speech_conditioning_input, first_inputs, second_inputs], dim=1)
        else:
            emb = torch.cat([speech_conditioning_input, first_inputs], dim=1)

        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        if get_attns:
            return gpt_out.attentions

        enc = gpt_out.last_hidden_state[:, 1:]  # The first logit is tied to the speech_conditioning_input
        first_logits = self.final_norm(enc[:, :first_inputs.shape[1]])
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0,2,1)
        if second_inputs is not None:
            second_logits = self.final_norm(enc[:, -second_inputs.shape[1]:])
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0,2,1)
            return first_logits, second_logits
        else:
            return first_logits

    def forward(self, speech_conditioning_input, text_inputs, mel_inputs, wav_lengths, text_first=True, return_attentions=False):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        speech_conditioning_input: MEL float tensor, (b,80,s)
        text_inputs: long tensor, (b,t)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        """
        mel_inputs = self.set_mel_padding(mel_inputs, wav_lengths)
        speech_conditioning_input = self.randomly_permute_conditioning_input(speech_conditioning_input)
        speech_conditioning_input = self.conditioning_encoder(speech_conditioning_input).unsqueeze(1)

        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.START_TEXT_TOKEN, self.STOP_TEXT_TOKEN)
        text_emb = self.text_embedding(text_inputs)
        mel_inputs, mel_targets = self.build_aligned_inputs_and_targets(mel_inputs, self.START_MEL_TOKEN, self.STOP_MEL_TOKEN)
        mel_emb = self.gpt.get_input_embeddings()(mel_inputs)
        if text_first:
            text_logits, mel_logits = self.get_logits(speech_conditioning_input, text_emb, self.text_head, mel_emb, self.mel_head, get_attns=return_attentions)
        else:
            mel_logits, text_logits = self.get_logits(speech_conditioning_input, mel_emb, self.mel_head, text_emb, self.text_head, get_attns=return_attentions)

        if return_attentions:
            return mel_logits
        loss_text = F.cross_entropy(text_logits, text_targets.long())
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long())
        return loss_text.mean(), loss_mel.mean(), mel_logits

    def text_forward(self, speech_conditioning_input, text_inputs):
        """
        Performs autoregressive modeling on only text. Still requires a speech_conditioning_input due to the way the
        model inputs are formatted. Just provide any audio clip (arguably, zeros could be provided).
        """
        speech_conditioning_input = self.randomly_permute_conditioning_input(speech_conditioning_input)
        speech_conditioning_input = self.conditioning_encoder(speech_conditioning_input).unsqueeze(1)

        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.START_TEXT_TOKEN, self.STOP_TEXT_TOKEN)
        text_emb = self.text_embedding(text_inputs)
        text_logits = self.get_logits(speech_conditioning_input, text_emb, self.text_head)
        loss_text = F.cross_entropy(text_logits, text_targets.long())
        return loss_text.mean()

    def speech_forward(self, speech_conditioning_input, mel_inputs, wav_lengths):
        """
        Performs autoregressive modeling on only speech data.
        """
        mel_inputs = self.set_mel_padding(mel_inputs, wav_lengths)
        speech_conditioning_input = self.randomly_permute_conditioning_input(speech_conditioning_input)
        speech_conditioning_input = self.conditioning_encoder(speech_conditioning_input).unsqueeze(1)

        mel_inputs, mel_targets = self.build_aligned_inputs_and_targets(mel_inputs, self.START_MEL_TOKEN, self.STOP_MEL_TOKEN)
        mel_emb = self.gpt.get_input_embeddings()(mel_inputs)
        mel_logits = self.get_logits(speech_conditioning_input, mel_emb, self.mel_head)
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long())
        return loss_mel.mean()

    def inference_speech(self, speech_conditioning_input, text_inputs, **hf_generate_kwargs):
        if not hasattr(self, 'inference_model'):
            self.inference_model = GPT2InferenceModel(self.gpt_config, self.gpt, None, self.final_norm, self.mel_head)

        text_inputs = F.pad(text_inputs, (0, self.max_symbols_per_phrase - text_inputs.shape[1]), value=self.STOP_TEXT_TOKEN)
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.START_TEXT_TOKEN, self.STOP_TEXT_TOKEN)
        text_emb = self.text_embedding(text_inputs)

        # Randomly permute the conditioning spectrogram, to destroy any structure present.
        speech_conditioning_input = self.randomly_permute_conditioning_input(speech_conditioning_input)
        cond = self.conditioning_encoder(speech_conditioning_input).unsqueeze(1)

        emb = torch.cat([cond, text_emb], dim=1)
        self.inference_model.store_mel_emb(emb)

        fake_inputs = torch.full((emb.shape[0],emb.shape[1]+1,), fill_value=1, dtype=torch.long, device=text_inputs.device)
        fake_inputs[:,-1] = self.START_MEL_TOKEN

        gen = self.inference_model.generate(fake_inputs, bos_token_id=self.START_MEL_TOKEN, pad_token_id=self.STOP_MEL_TOKEN, eos_token_id=self.STOP_MEL_TOKEN,
                                            max_length=emb.shape[1]+self.max_mel_tokens, **hf_generate_kwargs)
        return gen[:, fake_inputs.shape[1]:]


@register_model
def register_unified_gpt_voice(opt_net, opt):
    return UnifiedGptVoice(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    gpt = UnifiedGptVoice(model_dim=256, heads=4)
    l = gpt(torch.randn(2, 80, 800),
            torch.randint(high=len(symbols), size=(2,80)),
            torch.randint(high=8192, size=(2,250)),
            torch.tensor([150*256,195*256]))
