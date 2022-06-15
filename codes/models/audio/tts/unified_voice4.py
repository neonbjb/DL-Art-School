import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from models.arch_util import AttentionBlock
from models.audio.tts.transformer_builders import build_hf_gpt_transformer
from models.lucidrains.x_transformers import RotaryEmbedding, apply_rotary_pos_emb
from trainer.networks import register_model
from utils.util import opt_get


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


class GPT2InferenceModel(GPT2PreTrainedModel):
    def __init__(self, config, gpt, posterior_pos_emb, embeddings, norm, linear):
        super().__init__(config)
        self.transformer = gpt
        self.posterior_pos_embedding = posterior_pos_emb
        self.embeddings = embeddings
        self.head = nn.Sequential(norm, linear)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.cached_prior_emb = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.head = self.head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.head = self.head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def store_prior_emb(self, mel_emb):
        self.cached_prior_emb = mel_emb

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
        assert self.cached_prior_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Create embedding
        prior_len = self.cached_prior_emb.shape[1]
        if input_ids.shape[1] != 1:
            posterior_inputs = input_ids[:, prior_len:]
            posterior_emb = self.embeddings(posterior_inputs)
            posterior_emb = posterior_emb + self.posterior_pos_embedding(posterior_emb)
            if self.cached_prior_emb.shape[0] != posterior_emb.shape[0]:
                prior_emb = self.cached_prior_emb.repeat_interleave(posterior_emb.shape[0] // self.cached_prior_emb.shape[0], 0)
            else:
                prior_emb = self.cached_prior_emb
            emb = torch.cat([prior_emb, posterior_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.posterior_pos_embedding.get_fixed_embedding(attention_mask.shape[1] - prior_len, attention_mask.device)

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
            hidden_states = hidden_states.to(self.head.weight.device)

        logits = self.head(hidden_states)

        if not return_dict:
            return (logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=logits,
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


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False,
                 mean=False):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads, do_checkpoint=do_checkpointing))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        if self.mean:
            return h.mean(dim=2)
        else:
            return h[:, :, 0]


class MelEncoder(nn.Module):
    def __init__(self, channels, mel_channels=80, resblocks_per_reduction=2):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(nn.Conv1d(mel_channels, channels//4, kernel_size=3, padding=1),
                                     nn.Sequential(*[ResBlock(channels//4) for _ in range(resblocks_per_reduction)]),
                                     nn.Conv1d(channels//4, channels//2, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels//16, channels//2),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels//2) for _ in range(resblocks_per_reduction)]),
                                     nn.Conv1d(channels//2, channels, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels//8, channels),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels) for _ in range(resblocks_per_reduction)]),
                                     )
        self.reduction = 4


    def forward(self, x):
        for e in self.encoder:
            x = e(x)
        return x.permute(0,2,1)


class UnifiedVoice(nn.Module):
    def __init__(self, layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250, max_conditioning_inputs=1,
                 mel_length_compression=1024, number_text_tokens=256, number_mel_codes=8194, start_mel_token=8192,
                 stop_mel_token=8193, start_text_token=255, checkpointing=True, types=1, only_alignment_head=False):
        super().__init__()

        self.number_text_tokens = number_text_tokens
        self.start_text_token = number_text_tokens * types if start_text_token is None else start_text_token
        self.stop_text_token = 0
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_conditioning_inputs = max_conditioning_inputs
        self.max_mel_tokens = -1 if max_mel_tokens == -1 else max_mel_tokens+2+self.max_conditioning_inputs
        self.max_text_tokens = -1 if max_text_tokens == -1 else max_text_tokens+2
        self.model_dim = model_dim
        self.mel_length_compression = mel_length_compression
        self.conditioning_encoder = ConditioningEncoder(80, model_dim, num_attn_heads=heads)
        self.text_embedding = nn.Embedding(self.number_text_tokens*types+1, model_dim)
        self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        self.gpt, self.mel_pos_embedding, self.text_pos_embedding, self.mel_layer_pos_embedding, self.text_layer_pos_embedding = \
            build_hf_gpt_transformer(layers, model_dim, heads, self.max_mel_tokens, self.max_text_tokens, checkpointing)

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens*types+1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)
        self.alignment_head = nn.Linear(model_dim, 256)

        if only_alignment_head:
            for p in self.parameters():
                p.DO_NOT_TRAIN = True
                p.requires_grad = False
            for p in self.alignment_head.parameters():
                del p.DO_NOT_TRAIN
                p.requires_grad = True

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding, self.mel_embedding]
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=.02)

    def get_grad_norm_parameter_groups(self):
        return {
            'conditioning_encoder': list(self.conditioning_encoder.parameters()),
            'gpt': list(self.gpt.parameters()),
            'heads': list(self.text_head.parameters()) + list(self.mel_head.parameters()),
        }

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
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def get_logits(self, speech_conditioning_inputs, first_inputs, second_inputs, return_latent=False):
        emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)

        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True)

        enc = gpt_out.last_hidden_state[:, 1:]  # The first logit is tied to the speech_conditioning_input
        enc = self.final_norm(enc)

        if return_latent:
            return enc[:, speech_conditioning_inputs.shape[1]:speech_conditioning_inputs.shape[1]+first_inputs.shape[1]], enc[:, -second_inputs.shape[1]:]

        text_logits = enc[:, :first_inputs.shape[1]]
        text_logits = self.text_head(text_logits)
        text_logits = text_logits.permute(0,2,1)

        mel_logits = enc[:, -second_inputs.shape[1]:]
        mel_logits = self.mel_head(mel_logits)
        mel_logits = mel_logits.permute(0,2,1)

        alignment_logits = enc[:, -second_inputs.shape[1]:]
        alignment_logits = self.alignment_head(alignment_logits)
        alignment_logits = alignment_logits.permute(0,2,1)

        return text_logits, mel_logits, alignment_logits


    def get_conditioning_latent(self, speech_conditioning_input):
        speech_conditioning_input = speech_conditioning_input.unsqueeze(1) if len(speech_conditioning_input.shape) == 3 else speech_conditioning_input
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
        conds = torch.stack(conds, dim=1)
        conds = conds.mean(dim=1).unsqueeze(1)
        return conds


    def forward(self, speech_conditioning_input, text_inputs, text_lengths, mel_codes, ctc_codes, wav_lengths, types=None, return_latent=False):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        speech_conditioning_input: MEL float tensor, (b,80,s)
        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)

        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """
        # Types are expressed by expanding the text embedding space.
        if types is not None:
            text_inputs = text_inputs * (1+types).unsqueeze(-1)

        # TODO: do this in the dataloader.
        for b in range(ctc_codes.shape[0]):
            last_code = 0
            for j in range(ctc_codes.shape[1]):
                if ctc_codes[b][j] == 0:
                    ctc_codes[b][j] = last_code
                else:
                    last_code = ctc_codes[b][j]
        alignment_targets = F.interpolate(ctc_codes.unsqueeze(1).float(), size=(mel_codes.shape[-1],), mode='nearest').long().squeeze()

        mel_codes = self.set_mel_padding(mel_codes, wav_lengths)
        text_inputs = F.pad(text_inputs, (0,1), value=self.stop_text_token)
        mel_codes = F.pad(mel_codes, (0,1), value=self.stop_mel_token)
        alignment_targets = F.pad(alignment_targets, (0,2), value=0)

        conds = self.get_conditioning_latent(speech_conditioning_input)
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)
        mel_inp = mel_codes
        mel_emb = self.mel_embedding(mel_inp)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        text_logits, mel_logits, alignment_logits = self.get_logits(conds, text_emb, mel_emb, return_latent=return_latent)
        if return_latent:
            return mel_logits[:, :-2]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.

        loss_text = F.cross_entropy(text_logits, text_targets.long())
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long())
        loss_alignment = F.cross_entropy(alignment_logits, alignment_targets)
        return loss_text.mean(), loss_mel.mean(), loss_alignment, mel_logits

    def inference_speech(self, speech_conditioning_input, text_inputs, **hf_generate_kwargs):
        if self.max_mel_tokens == -1:  # Assume if this is the case, max_mel_tokens=-1 also
            seq_length = 2002  # Arbitrary default.
        else:
            seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        if not hasattr(self, 'inference_model'):
            # TODO: Decouple gpt_config from this inference model.
            gpt_config = GPT2Config(vocab_size=self.max_mel_tokens,
                                    n_positions=seq_length,
                                    n_ctx=seq_length,
                                    n_embd=self.model_dim,
                                    n_layer=self.layers,
                                    n_head=self.heads,
                                    gradient_checkpointing=False,
                                    use_cache=True)
            self.inference_model = GPT2InferenceModel(gpt_config, self.gpt, self.mel_pos_embedding, self.mel_embedding, self.final_norm, self.mel_head)
            self.gpt.wte = self.mel_embedding

        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        speech_conditioning_input = speech_conditioning_input.unsqueeze(1) if len(speech_conditioning_input.shape) == 3 else speech_conditioning_input
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
        conds = torch.stack(conds, dim=1)
        conds = conds.mean(dim=1).unsqueeze(1)

        emb = torch.cat([conds, text_emb], dim=1)
        self.inference_model.store_prior_emb(emb)

        fake_inputs = torch.full((emb.shape[0], conds.shape[1]+emb.shape[1],), fill_value=1, dtype=torch.long, device=text_inputs.device)
        fake_inputs[:,-1] = self.start_mel_token

        gen = self.inference_model.generate(fake_inputs, bos_token_id=self.start_mel_token, pad_token_id=self.stop_mel_token, eos_token_id=self.stop_mel_token,
                                            max_length=seq_length, return_dict_in_generate=True, **hf_generate_kwargs)
        return gen.sequences[:, fake_inputs.shape[1]:]


@register_model
def register_unified_voice4(opt_net, opt):
    return UnifiedVoice(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    gpt = UnifiedVoice(model_dim=256, heads=4, max_conditioning_inputs=4, types=2)
    l = gpt(torch.randn(2, 3, 80, 800),
            torch.randint(high=256, size=(2,120)),
            torch.tensor([32, 120]),
            torch.randint(high=8192, size=(2,250)),
            torch.tensor([250*256,195*256]),
            types=torch.tensor([0, 1]))
