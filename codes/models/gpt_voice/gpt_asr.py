from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import munchify

from models.gpt_voice.lucidrains_gpt import Transformer
from models.tacotron2.taco_utils import get_mask_from_lengths
from models.tacotron2.text import symbols, sequence_to_text
from trainer.networks import register_model
from utils.util import opt_get


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=5, padding = 2),
            nn.BatchNorm1d(chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=5, padding = 2),
            nn.BatchNorm1d(chan)
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class MelEncoder(nn.Module):
    def __init__(self, channels, mel_channels=80):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(nn.Conv1d(mel_channels, channels//4, kernel_size=7, padding=3),
                                     ResBlock(channels//4),
                                     ResBlock(channels//4),
                                     nn.Conv1d(channels//4, channels//2, kernel_size=5, stride=2, padding=2),
                                     nn.BatchNorm1d(channels//2),
                                     nn.ReLU(),
                                     ResBlock(channels//2),
                                     ResBlock(channels//2),
                                     ResBlock(channels//2),
                                     nn.Conv1d(channels//2, channels, kernel_size=5, stride=2, padding=2),
                                     ResBlock(channels),
                                     ResBlock(channels),
                                     ResBlock(channels)
                                     )

    def forward(self, x):
        return self.encoder(x)


class GptAsr(nn.Module):
    NUMBER_SYMBOLS = len(symbols)
    NUMBER_TEXT_TOKENS = NUMBER_SYMBOLS+1

    def __init__(self, layers=8, model_dim=512, heads=8, max_symbols_per_phrase=200, max_mel_frames=1000):
        super().__init__()
        self.max_mel_frames = max_mel_frames // 4  # Mel frames are reduced by a factor of 4 during encoding.
        self.max_symbols_per_phrase = max_symbols_per_phrase

        self.model_dim = model_dim
        self.max_mel_frames = self.max_mel_frames
        self.text_embedding = nn.Embedding(self.NUMBER_TEXT_TOKENS, model_dim)
        self.mel_encoder = MelEncoder(model_dim)
        self.text_pos_embedding = nn.Embedding(self.max_symbols_per_phrase + 1, model_dim)
        self.mel_pos_embedding = nn.Embedding(self.max_mel_frames, model_dim)
        self.gpt = Transformer(dim=model_dim, depth=layers, seq_len=2 + self.max_symbols_per_phrase + self.max_mel_frames, heads=heads,
                               attn_dropout=.1, ff_dropout=.1, non_causal_sequence_partition=self.max_mel_frames)

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.NUMBER_TEXT_TOKENS)

    def get_logits(self, mel_inputs, text_targets):
        # Pad front and back. Pad at front is the "START" token.
        text_targets = F.pad(text_targets, (1,0), value=self.NUMBER_SYMBOLS)
        text_targets = F.pad(text_targets, (0, self.max_symbols_per_phrase - text_targets.shape[1]))
        text_emb = self.text_embedding(text_targets)
        text_emb = text_emb + self.text_pos_embedding(torch.arange(text_emb.shape[1], device=text_targets.device))
        mel_emb = self.mel_encoder(mel_inputs)
        mel_emb = F.pad(mel_emb, (0, self.max_mel_frames - mel_emb.shape[-1]))
        mel_emb = mel_emb.permute(0,2,1).contiguous()
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_emb.device))
        emb = torch.cat([mel_emb, text_emb], dim=1)
        enc = self.gpt(emb)

        text_logits = self.final_norm(enc[:, self.max_mel_frames:])
        text_logits = self.text_head(text_logits)
        text_logits = text_logits.permute(0,2,1)
        return text_logits

    def forward(self, mel_inputs, text_targets):
        text_logits = self.get_logits(mel_inputs, text_targets)
        loss_text = F.cross_entropy(text_logits[:,:,:-1], text_targets[:,1:].long())
        return loss_text.mean(), text_logits

    def inference_beam_topk(self, mel, fn='inference_beam'):
        def topk_sampler(distribution, k):
            return torch.topk(distribution, k=k, dim=-1)
        return getattr(self, fn)(mel, topk_sampler)

    def inference_beam_sampled(self, mel, fn='inference_beam'):
        def multinomial_sampler(distribution, k):
            indices = torch.multinomial(distribution, num_samples=k, replacement=False)
            values = torch.gather(distribution, dim=1, index=indices)
            class container:
                def __init__(self, i, v):
                    self.indices = i
                    self.values = v
            return container(indices, values)
        return getattr(self, fn)(mel, multinomial_sampler)

    def inference_beam(self, mel_inputs, sampler_fn):
        beam_width = 16
        temperature = .8

        b, _, s = mel_inputs.shape
        assert b == 1  # Beam search only works on batches of one.
        mel_emb = self.mel_encoder(mel_inputs)
        mel_emb = F.pad(mel_emb, (0, self.max_mel_frames - mel_emb.shape[-1]))
        mel_emb = mel_emb.permute(0,2,1).contiguous()
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_emb.device))

        text_seq = torch.full((b,1), fill_value=self.NUMBER_SYMBOLS, device=mel_emb.device)
        probabilities = torch.ones((b,), device=mel_emb.device)
        while text_seq.shape[-1] < self.max_symbols_per_phrase:
            text_emb = self.text_embedding(text_seq)
            text_emb = text_emb + self.text_pos_embedding(torch.arange(text_emb.shape[1], device=mel_emb.device))
            if text_emb.shape[0] != mel_emb.shape[0]:
                mel_emb = mel_emb.repeat(text_emb.shape[0], 1, 1)
            emb = torch.cat([mel_emb, text_emb], dim=1)
            enc = self.gpt(emb)
            text_logits = self.final_norm(enc[:, mel_emb.shape[1]:])
            text_logits = self.text_head(text_logits)
            topk = sampler_fn(F.softmax(temperature * text_logits[:, -1], dim=-1), k=beam_width)
            probabilities = (probabilities.repeat_interleave(beam_width, dim=0) * topk.values.flatten())
            probabilities, sort_indices = torch.sort(probabilities, descending=True)
            probabilities = probabilities[:beam_width]

            text_seq = text_seq.repeat_interleave(beam_width, dim=0)
            codes = topk.indices.flatten()
            text_seq = torch.cat([text_seq, codes.unsqueeze(1)], dim=1)
            text_seq = text_seq[sort_indices]
            text_seq = text_seq[:beam_width]

            # PAD doubles as a stop token. PAD=0.
            if torch.all(torch.any(text_seq == 0, dim=1)):
                break

        if text_seq.shape[1] >= self.max_mel_frames:
            print("Warning! Encountered frame limit before a pad token. Output is likely wrong.")

        return text_seq


    def inference_beam_opt(self, mel_inputs, sampler_fn):
        beam_width = 16
        temperature = .8

        b, _, s = mel_inputs.shape
        assert b == 1  # Beam search only works on batches of one.
        mel_emb = self.mel_encoder(mel_inputs)
        mel_emb = F.pad(mel_emb, (0, self.max_mel_frames - mel_emb.shape[-1]))
        mel_emb = mel_emb.permute(0,2,1).contiguous()
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_emb.device))

        intermediates = []
        text_seq = torch.full((b,1), fill_value=self.NUMBER_SYMBOLS, device=mel_emb.device)
        probabilities = torch.ones((b,), device=mel_emb.device)
        while text_seq.shape[-1] < self.max_symbols_per_phrase:
            text_emb = self.text_embedding(text_seq)
            text_emb = text_emb + self.text_pos_embedding(torch.arange(text_emb.shape[1], device=mel_emb.device))
            if text_emb.shape[0] != mel_emb.shape[0]:
                mel_emb = mel_emb.repeat(text_emb.shape[0], 1, 1)
            emb = torch.cat([mel_emb, text_emb], dim=1)

            if len(intermediates) == 0:
                enc, intermediates = self.gpt(emb, return_intermediates=True)
                intermediates = [(i[0].repeat(beam_width, 1, 1),
                                  i[1].repeat(beam_width, 1, 1)) for i in intermediates]
            else:
                enc, intermediates = self.gpt.infer_last_two(emb, intermediates)

            text_logits = self.final_norm(enc[:, mel_emb.shape[1]:])
            text_logits = self.text_head(text_logits)
            topk = sampler_fn(F.softmax(temperature * text_logits[:, -1], dim=-1), k=beam_width)
            probabilities = (probabilities.repeat_interleave(beam_width, dim=0) * topk.values.flatten())
            probabilities, sort_indices = torch.sort(probabilities, descending=True)
            probabilities = probabilities[:beam_width]

            text_seq = text_seq.repeat_interleave(beam_width, dim=0)
            codes = topk.indices.flatten()
            text_seq = torch.cat([text_seq, codes.unsqueeze(1)], dim=1)
            text_seq = text_seq[sort_indices]
            text_seq = text_seq[:beam_width]

            # PAD doubles as a stop token. PAD=0.
            if torch.all(torch.any(text_seq == 0, dim=1)):
                break

        if text_seq.shape[1] >= self.max_mel_frames:
            print("Warning! Encountered frame limit before a pad token. Output is likely wrong.")

        return text_seq


@register_model
def register_gpt_asr(opt_net, opt):
    return GptAsr(**opt_get(opt_net, ['kwargs'], {}))


# Quick script that loads a model and halves the number of layers, then saves that model.
def distill():
    gpt = GptAsr(max_symbols_per_phrase=250, max_mel_frames=1400, layers=12, model_dim=768, heads=12)
    gpt.load_state_dict(torch.load('../experiments/train_gpt_asr_mass/models/21500_mel_gen.pth'))
    rc = 0
    i = 0
    while i < len(gpt.gpt.layers.layers):
        if rc % 2 != 0:
            del gpt.gpt.layers.layers[i]
        else:
            i += 1
        rc += 1
    torch.save(gpt.state_dict(), '../experiments/train_gpt_asr_mass/models/21500_mel_gen_distilled.pth')


if __name__ == '__main__':
    gpt = GptAsr(max_symbols_per_phrase=100, max_mel_frames=200, layers=6, model_dim=256, heads=2).cuda()
    #l = gpt(torch.randn(2,80,800),
    #           torch.randint(high=len(symbols), size=(2,180)))

    with torch.no_grad():
        t = torch.randn(1,80,800).cuda()
        start = time()
        s = gpt.inference_beam_topk(t)
        print(time()-start)

        start = time()
        o = gpt.inference_beam_topk(t, fn='inference_beam_opt')
        print(time()-start)


