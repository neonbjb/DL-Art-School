import functools
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import T5Config, T5Model, T5PreTrainedModel, T5ForConditionalGeneration
from transformers.file_utils import replace_return_docstrings
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
from x_transformers import Encoder, XTransformer

from models.gpt_voice.transformer_builders import null_position_embeddings
from models.gpt_voice.unet_diffusion_tts6 import CheckpointedLayer
from models.gpt_voice.unified_voice2 import ConditioningEncoder
from models.tacotron2.text.cleaners import english_cleaners
from trainer.networks import register_model
from utils.util import opt_get


class CtcCodeGenerator(nn.Module):
    def __init__(self, model_dim=512, layers=10, num_heads=8, dropout=.1, ctc_codes=36, max_pad=121, max_repeat=30, checkpointing=True):
        super().__init__()
        self.max_pad = max_pad
        self.max_repeat = max_repeat
        self.start_token = self.max_repeat*self.max_pad+1
        self.conditioning_encoder = ConditioningEncoder(80, model_dim, num_attn_heads=num_heads)
        self.embedding = nn.Embedding(ctc_codes, model_dim)
        self.config = T5Config(
            vocab_size=self.start_token+1,
            d_model=model_dim,
            d_kv=model_dim//num_heads,
            d_ff=model_dim*4,
            num_layers=layers,
            num_heads=num_heads,
            dropout_rate=dropout,
            feed_forward_proj='gated-gelu',
            use_cache=not checkpointing,
            gradient_checkpointing=checkpointing,
            tie_word_embeddings=False,
            tie_encoder_decoder=False,
            decoder_start_token_id=self.start_token,
            pad_token_id=0,
        )
        self.transformer = T5ForConditionalGeneration(self.config)
        del self.transformer.encoder.embed_tokens
        del self.transformer.shared
        self.transformer.encoder.embed_tokens = functools.partial(null_position_embeddings, dim=model_dim)

    def forward(self, conditioning_input, codes, separators, repeats, unpadded_lengths):
        max_len = unpadded_lengths.max()
        codes = codes[:, :max_len]
        separators = separators[:, :max_len]
        repeats = repeats[:, :max_len]
        if separators.max() > self.max_pad:
            print(f"Got unexpectedly long separators. Max: {separators.max()}, {separators}")
            separators = torch.clip(separators, 0, self.max_pad)
        if repeats.max() > self.max_repeat:
            print(f"Got unexpectedly long repeats. Max: {repeats.max()}, {repeats}")
            repeats = torch.clip(repeats, 0, self.max_repeat)
        assert not torch.any(repeats < 1)
        repeats = repeats - 1  # Per above, min(repeats) is 1; make it 0 to avoid wasting a prediction slot.

        assert codes.max() < 36, codes.max()
        labels = separators + repeats * self.max_pad
        labels = labels + 1  # We want '0' to be used as the EOS or padding token, so add 1.
        for i in range(unpadded_lengths.shape[0]):
            labels[i, unpadded_lengths[i]:] = 0

        conditioning_input = conditioning_input.unsqueeze(1) if len(conditioning_input.shape) == 3 else conditioning_input
        conds = []
        for j in range(conditioning_input.shape[1]):
            conds.append(self.conditioning_encoder(conditioning_input[:, j]))
        conds = torch.stack(conds, dim=1)
        h = torch.cat([conds, self.embedding(codes)], dim=1)

        decoder_inputs = F.pad(labels, (1, 0), value=self.start_token)[:, :-1]
        loss = self.transformer(inputs_embeds=h, decoder_input_ids=decoder_inputs, labels=labels).loss
        return loss

    def generate(self, speech_conditioning_inputs, texts, **hf_generate_kwargs):
        codes = []
        max_seq = 50
        for text in texts:
            # First, generate CTC codes from the given texts.
            vocab = json.loads('{" ": 4, "E": 5, "T": 6, "A": 7, "O": 8, "N": 9, "I": 10, "H": 11, "S": 12, "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18, "C": 19, "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26, "\'": 27, "X": 28, "J": 29, "Q": 30, "Z": 31}')
            text = english_cleaners(text)
            text = text.strip().upper()
            cd = []
            for c in text:
                if c not in vocab.keys():
                    continue
                cd.append(vocab[c])
            codes.append(torch.tensor(cd, device=speech_conditioning_inputs.device))
            max_seq = max(max_seq, codes[-1].shape[-1])
        # Collate
        for i in range(len(codes)):
            if codes[i].shape[-1] < max_seq:
                codes[i] = F.pad(codes[i], (0, max_seq-codes[i].shape[-1]))
        codes = torch.stack(codes, dim=0)

        conditioning_input = speech_conditioning_inputs.unsqueeze(1) if len(speech_conditioning_inputs.shape) == 3 else speech_conditioning_inputs
        conds = []
        for j in range(conditioning_input.shape[1]):
            conds.append(self.conditioning_encoder(conditioning_input[:, j]))
        conds = torch.stack(conds, dim=1)
        h = torch.cat([conds, self.embedding(codes)], dim=1)
        generate = self.transformer.generate(inputs_embeds=h, max_length=codes.shape[-1]+1, min_length=codes.shape[-1]+1,
                                             bos_token_id=self.start_token,
                                             bad_words_ids=[[0], [self.start_token]], **hf_generate_kwargs)
        # The HF generate API returns a sequence with the BOS token included, hence the +1s above. Remove it.
        generate = generate[:, 1:]

        # De-compress the codes from the generated output
        generate = generate - 1  # Remember above when we added 1 to the labels to avoid overlapping the EOS pad token?
        pads = generate % self.max_pad
        repeats = (generate // self.max_pad) + 1
        ctc_batch = []
        max_seq = 0
        for bc, bp, br in zip(codes, pads, repeats):
            ctc = []
            for c, p, r in zip(bc, bp, br):
                for _ in range(p):
                    ctc.append(0)
                for _ in range(r):
                    ctc.append(c.item())
            ctc_batch.append(torch.tensor(ctc, device=speech_conditioning_inputs.device))
            max_seq = max(max_seq, ctc_batch[-1].shape[-1])

        # Collate the batch
        for i in range(len(ctc_batch)):
            if ctc_batch[i].shape[-1] < max_seq:
                ctc_batch[i] = F.pad(ctc_batch[i], (0, max_seq-ctc_batch[i].shape[-1]))
        return torch.stack(ctc_batch, dim=0)


@register_model
def register_ctc_code_generator2(opt_net, opt):
    return CtcCodeGenerator(**opt_get(opt_net, ['kwargs'], {}))


def inf():
    sd = torch.load('D:\\dlas\\experiments\\train_encoder_build_ctc_alignments\\models\\24000_generator.pth', map_location='cpu')
    model = CtcCodeGenerator(layers=10, checkpointing=False).eval()
    model.load_state_dict(sd)
    raw_batch = torch.load('raw_batch.pth')
    with torch.no_grad():
        from data.audio.unsupervised_audio_dataset import load_audio
        from scripts.audio.gen.speech_synthesis_utils import wav_to_mel
        ref_mel = torch.cat([wav_to_mel(raw_batch['conditioning'][0])[:, :, :256],
                               wav_to_mel(raw_batch['conditioning'][0])[:, :, :256]], dim=0).unsqueeze(0)
        loss = model(ref_mel, raw_batch['ctc_raw_codes'][0].unsqueeze(0),
                     raw_batch['ctc_pads'][0].unsqueeze(0),
                     raw_batch['ctc_repeats'][0].unsqueeze(0),
                     raw_batch['ctc_raw_lengths'][0].unsqueeze(0),)
        #ref_mel = torch.cat([wav_to_mel(load_audio("D:\\tortoise-tts\\voices\\atkins\\1.wav", 22050))[:, :, :256],
        #                       wav_to_mel(load_audio("D:\\tortoise-tts\\voices\\atkins\\2.wav", 22050))[:, :, :256]], dim=0).unsqueeze(0)
        #ctc = model.generate(ref_mel, ["i suppose though it's too early for them"], num_beams=4, )
    print("Break")


if __name__ == '__main__':
    inf()

    model = CtcCodeGenerator()
    conds = torch.randn(4,2,80,600)
    inps = torch.randint(0,36, (4, 300))
    pads = torch.randint(0,100, (4,300))
    repeats = torch.randint(0,20, (4,300))
    #loss = model(conds, inps, pads, repeats, torch.tensor([250, 300, 280, 30]))
    #print(loss.shape)
    #model.generate(conds, ["Hello, world!", "Ahoi!", "KKKKKK", "what's going on??"])