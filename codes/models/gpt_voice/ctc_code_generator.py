import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder, XTransformer, TransformerWrapper

from models.gpt_voice.unet_diffusion_tts6 import CheckpointedLayer
from models.gpt_voice.unified_voice2 import ConditioningEncoder
from models.tacotron2.text.cleaners import english_cleaners
from trainer.networks import register_model
from utils.util import opt_get


class CheckpointedTransformerWrapper(nn.Module):
    """
    Wraps a ContinuousTransformerWrapper and applies CheckpointedLayer to each layer and permutes from channels-mid
    to channels-last that XTransformer expects.
    """
    def __init__(self, **xtransformer_kwargs):
        super().__init__()
        self.transformer = TransformerWrapper(**xtransformer_kwargs)

        for i in range(len(self.transformer.transformer.attn_layers.layers)):
            n, b, r = self.transformer.transformer.attn_layers.layers[i]
            self.transformer.transformer.attn_layers.layers[i] = nn.ModuleList([n, CheckpointedLayer(b), r])

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)


class CtcCodeGenerator(nn.Module):
    def __init__(self, model_dim=512, layers=10, num_heads=8, dropout=.1, ctc_codes=36, max_pad=121, max_repeat=30):
        super().__init__()
        self.max_pad = max_pad
        self.max_repeat = max_repeat
        self.conditioning_encoder = ConditioningEncoder(80, model_dim, num_attn_heads=num_heads)
        self.initial_embedding = nn.Embedding(ctc_codes, model_dim)
        self.combiner = nn.Linear(model_dim*2, model_dim)
        self.transformer = TransformerWrapper(
            num_tokens=max_pad*max_repeat+1,
            max_seq_len=-1,
            attn_layers=Encoder(
                dim=model_dim,
                depth=layers,
                heads=num_heads,
                ff_dropout=dropout,
                attn_dropout=dropout,
                use_rmsnorm=True,
                ff_glu=True,
                rotary_pos_emb=True
            )
        )
        self.transformer.token_emb = nn.Identity()  # This class handles the initial embeddings.

    def forward(self, conditioning_input, codes, separators, repeats, unpadded_lengths):
        max_len = unpadded_lengths.max()
        codes = codes[:, :max_len]
        if separators.max() > self.max_pad:
            print(f"Got unexpectedly long separators. Max: {separators.max()}, {separators}")
            separators = torch.clip(separators, 0, self.max_pad)
        separators = separators[:, :max_len]
        if repeats.max() > self.max_repeat:
            print(f"Got unexpectedly long repeats. Max: {repeats.max()}, {repeats}")
            repeats = torch.clip(repeats, 1, self.max_repeat)
        repeats = repeats[:, :max_len]
        repeats = repeats - 1  # min(repeats) is 1; make it 0 to avoid wasting a prediction slot.
        labels = separators + repeats * self.max_pad
        labels = labels + 1  # We want '0' to be used as the EOS or padding token, so add 1.
        for i in range(unpadded_lengths.shape[0]):
            labels[i, unpadded_lengths[i]:] = 0

        cond = self.conditioning_encoder(conditioning_input).unsqueeze(1).repeat(1,codes.shape[1],1)
        h = torch.cat([cond, self.initial_embedding(codes)], dim=-1)
        h = self.combiner(h)
        logits = self.transformer(h)
        loss = F.cross_entropy(logits.permute(0,2,1), labels)
        return loss

    def generate(self, speech_conditioning_input, texts):
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
            codes.append(torch.tensor(cd, device=speech_conditioning_input.device))
            max_seq = max(max_seq, codes[-1].shape[-1])
        # Collate
        for i in range(len(codes)):
            if codes[i].shape[-1] < max_seq:
                codes[i] = F.pad(codes[i], (0, max_seq-codes[i].shape[-1]))
        codes = torch.stack(codes, dim=0)

        cond = self.conditioning_encoder(speech_conditioning_input).unsqueeze(1).repeat(1,codes.shape[1],1)
        h = torch.cat([cond, self.initial_embedding(codes)], dim=-1)
        h = self.combiner(h)
        logits = self.transformer(h)
        generate = torch.argmax(logits, dim=-1)

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
            ctc_batch.append(torch.tensor(ctc, device=speech_conditioning_input.device))
            max_seq = max(max_seq, ctc_batch[-1].shape[-1])

        # Collate the batch
        for i in range(len(ctc_batch)):
            if ctc_batch[i].shape[-1] < max_seq:
                ctc_batch[i] = F.pad(ctc_batch[i], (0, max_seq-ctc_batch[i].shape[-1]))
        return torch.stack(ctc_batch, dim=0)

@register_model
def register_ctc_code_generator(opt_net, opt):
    return CtcCodeGenerator(**opt_get(opt_net, ['kwargs'], {}))


def inf():
    sd = torch.load('D:\\dlas\\experiments\\train_encoder_build_ctc_alignments\\models\\11000_generator.pth', map_location='cpu')
    model = CtcCodeGenerator(layers=10).eval()
    model.load_state_dict(sd)
    #raw_batch = torch.load('raw_batch.pth')
    with torch.no_grad():
        from data.audio.unsupervised_audio_dataset import load_audio
        from scripts.audio.gen.speech_synthesis_utils import wav_to_mel
        #loss = model(wav_to_mel(raw_batch['conditioning'][0]),
        #             raw_batch['ctc_raw_codes'][0].unsqueeze(0),
        #             raw_batch['ctc_pads'][0].unsqueeze(0),
        #             raw_batch['ctc_repeats'][0].unsqueeze(0),
        #             raw_batch['ctc_raw_lengths'][0].unsqueeze(0),)
        ref_mel = wav_to_mel(load_audio("D:\\tortoise-tts\\voices\\atkins\\1.wav", 22050))
        ctc = model.generate(ref_mel, ["i suppose though it's too early for them"])
    print("Break")


if __name__ == '__main__':
    inf()
    model = CtcCodeGenerator()
    inps = torch.randint(0,36, (4, 300))
    pads = torch.randint(0,100, (4,300))
    repeats = torch.randint(1,20, (4,300))
    conds = torch.randn(4,80,600)
    loss = model(conds, inps, pads, repeats, torch.tensor([250, 300, 280, 30]))
    print(loss.shape)