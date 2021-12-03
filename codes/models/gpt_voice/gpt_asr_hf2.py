from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from models.tacotron2.text import symbols
from trainer.networks import register_model
from utils.audio import plot_spectrogram
from utils.util import opt_get


class ResBlock(nn.Module):
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


class MelEncoder(nn.Module):
    def __init__(self, channels, mel_channels=80):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(nn.Conv1d(mel_channels, channels//4, kernel_size=3, padding=1),
                                     ResBlock(channels//4),
                                     nn.Conv1d(channels//4, channels//2, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels//16, channels//2),
                                     nn.ReLU(),
                                     ResBlock(channels//2),
                                     nn.Conv1d(channels//2, channels, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels//8, channels),
                                     nn.ReLU(),
                                     ResBlock(channels)
                                     )

    def forward(self, x):
        for e in self.encoder:
            x = e(x)
        return x


class GPT2InferenceModel(GPT2PreTrainedModel):
    def __init__(self, config, gpt, text_pos_emb, norm, linear):
        super().__init__(config)
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.lm_head = nn.Sequential(norm, linear)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.cached_mel_emb = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def store_mel_emb(self, mel_emb):
        self.cached_mel_emb = mel_emb

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):

        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert self.cached_mel_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Create embedding
        mel_len = self.cached_mel_emb.shape[1]
        if input_ids.shape[1] != 1:
            text_inputs = input_ids[:, mel_len:]
            text_emb = self.transformer.get_input_embeddings()(text_inputs)
            text_emb = text_emb + self.text_pos_embedding(torch.arange(text_emb.shape[1], device=text_emb.device))
            if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                mel_emb = self.cached_mel_emb.repeat_interleave(text_emb.shape[0]//self.cached_mel_emb.shape[0], 0)
            else:
                mel_emb = self.cached_mel_emb
            emb = torch.cat([mel_emb, text_emb], dim=1)
        else:
            emb = self.transformer.get_input_embeddings()(input_ids) + \
                          self.text_pos_embedding(torch.tensor(attention_mask.shape[1]-mel_len, device=attention_mask.device)).unsqueeze(0).unsqueeze(0)

        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class GptAsrHf2(nn.Module):
    NUMBER_SYMBOLS = len(symbols)
    START_TOKEN = NUMBER_SYMBOLS
    NUMBER_TEXT_TOKENS = NUMBER_SYMBOLS+2

    def __init__(self, layers=8, model_dim=512, heads=8, max_symbols_per_phrase=800, max_mel_frames=3000, checkpointing=True):
        super().__init__()
        self.max_mel_frames = max_mel_frames // 4  # Mel frames are reduced by a factor of 4 during encoding.
        self.max_symbols_per_phrase = max_symbols_per_phrase

        self.model_dim = model_dim
        self.max_mel_frames = self.max_mel_frames
        self.mel_encoder = MelEncoder(model_dim)
        self.text_pos_embedding = nn.Embedding(self.max_symbols_per_phrase + 1, model_dim)
        self.mel_pos_embedding = nn.Embedding(self.max_mel_frames, model_dim)
        seq_length = 2+self.max_symbols_per_phrase+self.max_mel_frames
        self.gpt_config = GPT2Config(vocab_size=self.NUMBER_TEXT_TOKENS,
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


    def get_logits(self, mel_inputs, text_targets, get_attns=False):
        # Pad front and remove last element to set up next token prediction. Pad at front is the "START" token.
        text_inputs = F.pad(text_targets, (1,0), value=self.START_TOKEN)[:, :-1]
        text_emb = self.gpt.get_input_embeddings()(text_inputs)
        text_emb = text_emb + self.text_pos_embedding(torch.arange(text_emb.shape[1], device=text_inputs.device))
        mel_emb = self.mel_encoder(mel_inputs)
        mel_emb = mel_emb.permute(0,2,1).contiguous()
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_emb.device))
        emb = torch.cat([mel_emb, text_emb], dim=1)
        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        if get_attns:
            return gpt_out.attentions
        enc = gpt_out.last_hidden_state
        text_logits = self.final_norm(enc[:, mel_emb.shape[1]:])
        text_logits = self.text_head(text_logits)
        text_logits = text_logits.permute(0,2,1)
        return text_logits

    def forward(self, mel_inputs, text_targets, return_attentions=False):
        text_targets = F.pad(text_targets, (0,1))  # Pad the targets with a <0> so that all have a "stop" token.
        text_logits = self.get_logits(mel_inputs, text_targets, get_attns=return_attentions)
        if return_attentions:
            return text_logits  # These weren't really the logits.
        loss_text = F.cross_entropy(text_logits, text_targets.long())
        return loss_text.mean(), text_logits

    def inference(self, mel_inputs, cond_text=None, do_sample=False, temperature=1.0, num_beams=8):
        if not hasattr(self, 'inference_model'):
            self.inference_model = GPT2InferenceModel(self.gpt_config, self.gpt, self.text_pos_embedding, self.final_norm, self.text_head)

        mel_emb = self.mel_encoder(mel_inputs)
        assert mel_emb.shape[-1] <= self.max_mel_frames
        mel_emb = mel_emb.permute(0,2,1).contiguous()
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_emb.device))
        self.inference_model.store_mel_emb(mel_emb)

        # "fake_inputs" are stand-ins for the MEL frames, which will be injected with the prep_inputs function above.
        if cond_text is None:
            fake_inputs = torch.full((mel_emb.shape[0],mel_emb.shape[1]+1,), fill_value=1, dtype=torch.long, device=mel_inputs.device)
            fake_inputs[:,-1] = self.START_TOKEN
        else:
            cond_used = 10
            fake_inputs = torch.full((mel_emb.shape[0],mel_emb.shape[1]+1+cond_used,), fill_value=1, dtype=torch.long, device=mel_inputs.device)
            fake_inputs[:,-1-cond_used] = self.START_TOKEN
            fake_inputs[:, -cond_used:] = cond_text[:, :cond_used]
        gen = self.inference_model.generate(fake_inputs, do_sample=do_sample, bos_token_id=self.START_TOKEN, pad_token_id=0, eos_token_id=0,
                          max_length=self.max_symbols_per_phrase+mel_emb.shape[1], temperature=temperature, num_beams=num_beams, use_cache=True)
        return gen[:, self.max_mel_frames:]


@register_model
def register_gpt_asr_hf2(opt_net, opt):
    return GptAsrHf2(**opt_get(opt_net, ['kwargs'], {}))


# Quick script that loads a model and halves the number of layers, then saves that model.
def distill():
    gpt = GptAsrHf2(max_symbols_per_phrase=250, max_mel_frames=1400, layers=12, model_dim=512, heads=8)
    gpt.load_state_dict(torch.load('X:\\dlas\\experiments\\train_gpt_asr_mass_hf\\models\\48000_gpt_ema.pth'))
    rc = 0
    i = 0
    while i < len(gpt.gpt.h):
        if rc % 2 != 0:
            del gpt.gpt.h[i]
        else:
            i += 1
        rc += 1
    torch.save(gpt.state_dict(), 'X:\\dlas\\experiments\\train_gpt_asr_mass_hf\\models\\48000_gpt_distilled.pth')


if __name__ == '__main__':
    #distill()

    gpt = GptAsrHf2(max_symbols_per_phrase=250, max_mel_frames=1400, layers=16, model_dim=512, heads=8)
    l = gpt(torch.randn(2,80,800), torch.randint(high=len(symbols), size=(2,100)))

    #start = time()
    #gpt.inference(torch.randn(1,80,350), num_beams=1)
    #print(f"Elapsed: {time()-start}")
