import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config

from models.arch_util import AttentionBlock
from models.gpt_voice.gpt_asr_hf import GPT2InferenceModel
from models.gpt_voice.gpt_asr_hf2 import ResBlock
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


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class UnifiedGptVoice(nn.Module):
    """
    Derived from GptTtsHf, but offers multiple modes of autoregressive operation:
    - Text only
    - Voice only
    - Text conditioned on voice
    - Voice conditioned on text
    """

    def __init__(self, layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250, max_conditioning_inputs=1,
                 max_conditioning_length=60, shuffle_conditioning=True, mel_length_compression=1024, number_text_tokens=256,
                 start_text_token=255, stop_text_token=0, number_mel_codes=8194, start_mel_token=8192,
                 stop_mel_token=8193, train_solo_embeddings=False, use_mel_codes_as_input=True,
                 checkpointing=True):
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            max_conditioning_length: Maximum length of conditioning input. Only needed if shuffle_conditioning=True
            shuffle_conditioning: Whether or not the conditioning inputs will be shuffled across the sequence dimension. Useful if you want to provide the same input as conditioning and mel_codes.
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
            number_text_tokens:
            start_text_token:
            stop_text_token:
            number_mel_codes:
            start_mel_token:
            stop_mel_token:
            train_solo_embeddings:
            use_mel_codes_as_input:
            checkpointing:
        """
        super().__init__()

        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.shuffle_conditioning = shuffle_conditioning

        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.conditioning_encoder = ConditioningEncoder(80, model_dim, num_attn_heads=heads)
        self.text_embedding = nn.Embedding(self.number_text_tokens, model_dim)
        self.text_pos_embedding = nn.Embedding(self.max_text_tokens + 1, model_dim)
        self.mel_pos_embedding = nn.Embedding(self.max_mel_tokens + 1, model_dim)
        seq_length = 2+max_text_tokens+self.max_mel_tokens+self.max_conditioning_inputs
        self.gpt_config = GPT2Config(vocab_size=self.number_mel_codes,
                                     n_positions=seq_length-100, # -100 is a hack for backwards compatibility. TODO: remove at some point.
                                     n_ctx=seq_length-100,
                                     n_embd=model_dim,
                                     n_layer=layers,
                                     n_head=heads,
                                     gradient_checkpointing=checkpointing,
                                     use_cache=not checkpointing)
        self.gpt = GPT2Model(self.gpt_config)
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * self.gpt.config.initializer_range, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * self.gpt.config.initializer_range, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0
        # Override the built in positional embeddings
        del self.gpt.wpe
        self.gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)

        if not use_mel_codes_as_input:
            self.gpt.wte = MelEncoder(model_dim, resblocks_per_reduction=1)

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)
        self.max_conditioning_length = max_conditioning_length

        # Initialize the embeddings per the GPT-2 scheme
        for module in [self.text_embedding, self.text_pos_embedding, self.mel_pos_embedding]:
            module.weight.data.normal_(mean=0.0, std=self.gpt.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

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
        enc = self.final_norm(enc)
        first_logits = enc[:, :first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0,2,1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1]:]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0,2,1)
            return first_logits, second_logits
        else:
            return first_logits

    def forward(self, speech_conditioning_input, text_inputs, mel_codes, wav_lengths, text_first=True, raw_mels=None, return_attentions=False):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        speech_conditioning_input: MEL float tensor, (b,80,s)
        text_inputs: long tensor, (b,t)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        raw_mels: MEL float tensor (b,80,s)
        """
        assert self.max_mel_tokens >= mel_codes.shape[1], f'{mel_codes.shape[1]}'
        assert self.max_text_tokens >= text_inputs.shape[1], f'{text_inputs.shape[1]}'

        mel_codes = self.set_mel_padding(mel_codes, wav_lengths)
        if self.shuffle_conditioning:
            speech_conditioning_input = self.randomly_permute_conditioning_input(speech_conditioning_input)
        speech_conditioning_input = self.conditioning_encoder(speech_conditioning_input).unsqueeze(1)

        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(torch.arange(text_inputs.shape[1], device=text_inputs.device))
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)
        if raw_mels is not None:
            mel_inp = F.pad(raw_mels, (0, 4))
        else:
            mel_inp = mel_codes
        mel_emb = self.gpt.get_input_embeddings()(mel_inp)
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_emb.device))
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
        assert self.max_text_tokens >= text_inputs.shape[1], f'{text_inputs.shape[1]}'

        if self.shuffle_conditioning:
            speech_conditioning_input = self.randomly_permute_conditioning_input(speech_conditioning_input)
        speech_conditioning_input = self.conditioning_encoder(speech_conditioning_input).unsqueeze(1)

        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(torch.arange(text_inputs.shape[1], device=text_inputs.device)) + self.text_solo_embedding
        text_logits = self.get_logits(speech_conditioning_input, text_emb, self.text_head)
        loss_text = F.cross_entropy(text_logits, text_targets.long())
        return loss_text.mean()

    def speech_forward(self, speech_conditioning_input, mel_codes, wav_lengths, raw_mels=None):
        """
        Performs autoregressive modeling on only speech data.
        """
        assert self.max_mel_tokens >= mel_codes.shape[1], f'{mel_codes.shape[1]}'

        mel_codes = self.set_mel_padding(mel_codes, wav_lengths)
        if self.shuffle_conditioning:
            speech_conditioning_input = self.randomly_permute_conditioning_input(speech_conditioning_input)
        speech_conditioning_input = self.conditioning_encoder(speech_conditioning_input).unsqueeze(1)

        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)
        if raw_mels is not None:
            mel_inp = F.pad(raw_mels, (0, 4))
        else:
            mel_inp = mel_codes
        mel_emb = self.gpt.get_input_embeddings()(mel_inp)
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_emb.device)) + self.mel_solo_embedding
        mel_logits = self.get_logits(speech_conditioning_input, mel_emb, self.mel_head)
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long())
        return loss_mel.mean()

    def inference_speech(self, speech_conditioning_input, text_inputs, **hf_generate_kwargs):
        if not hasattr(self, 'inference_model'):
            self.inference_model = GPT2InferenceModel(self.gpt_config, self.gpt, self.mel_pos_embedding, self.final_norm, self.mel_head)

        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(torch.arange(text_inputs.shape[1], device=text_inputs.device))

        if self.shuffle_conditioning:
            # Randomly permute the conditioning spectrogram, to destroy any structure present.
            speech_conditioning_input = self.randomly_permute_conditioning_input(speech_conditioning_input)
        cond = self.conditioning_encoder(speech_conditioning_input).unsqueeze(1)

        emb = torch.cat([cond, text_emb], dim=1)
        self.inference_model.store_mel_emb(emb)

        fake_inputs = torch.full((emb.shape[0],emb.shape[1]+1,), fill_value=1, dtype=torch.long, device=text_inputs.device)
        fake_inputs[:,-1] = self.start_mel_token

        gen = self.inference_model.generate(fake_inputs, bos_token_id=self.start_mel_token, pad_token_id=self.stop_mel_token, eos_token_id=self.stop_mel_token,
                                            max_length=self.gpt_config.n_positions, **hf_generate_kwargs)
        return gen[:, fake_inputs.shape[1]:]


@register_model
def register_unified_gpt_voice(opt_net, opt):
    return UnifiedGptVoice(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    gpt = UnifiedGptVoice(model_dim=256, heads=4, train_solo_embeddings=True, use_mel_codes_as_input=False)
    l = gpt(torch.randn(2, 80, 800),
            torch.randint(high=len(symbols), size=(2,80)),
            torch.randint(high=8192, size=(2,250)),
            torch.tensor([150*256,195*256]),
            raw_mels=torch.randn(2,80,1000))
    gpt.text_forward(torch.randn(2,80,800), torch.randint(high=50, size=(2,80)))
