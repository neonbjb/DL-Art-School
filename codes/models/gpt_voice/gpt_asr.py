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
    MAX_SYMBOLS_PER_PHRASE = 200
    MAX_MEL_FRAMES = 1000 // 4
    NUMBER_SYMBOLS = len(symbols)
    NUMBER_TEXT_TOKENS = NUMBER_SYMBOLS+1

    def __init__(self, layers=8, model_dim=512, heads=8):
        super().__init__()

        self.model_dim = model_dim
        self.max_mel_frames = self.MAX_MEL_FRAMES
        self.text_embedding = nn.Embedding(self.NUMBER_TEXT_TOKENS, model_dim)
        self.mel_encoder = MelEncoder(model_dim)
        self.text_pos_embedding = nn.Embedding(self.MAX_SYMBOLS_PER_PHRASE+1, model_dim)
        self.mel_pos_embedding = nn.Embedding(self.MAX_MEL_FRAMES, model_dim)
        self.gpt = Transformer(dim=model_dim, depth=layers, seq_len=2+self.MAX_SYMBOLS_PER_PHRASE+self.MAX_MEL_FRAMES, heads=heads,
                               attn_dropout=.1, ff_dropout=.1, non_causal_sequence_partition=self.MAX_MEL_FRAMES)

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.NUMBER_TEXT_TOKENS)

    def forward(self, mel_inputs, text_targets):
        # Pad front and back. Pad at front is the "START" token.
        text_targets = F.pad(text_targets, (1,0), value=self.NUMBER_SYMBOLS)
        text_targets = F.pad(text_targets, (0, self.MAX_SYMBOLS_PER_PHRASE-text_targets.shape[1]))
        text_emb = self.text_embedding(text_targets)
        text_emb = text_emb + self.text_pos_embedding(torch.arange(text_emb.shape[1], device=text_targets.device))
        mel_emb = self.mel_encoder(mel_inputs)
        mel_emb = F.pad(mel_emb, (0, self.MAX_MEL_FRAMES-mel_emb.shape[-1]))
        mel_emb = mel_emb.permute(0,2,1).contiguous()
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_emb.device))
        emb = torch.cat([mel_emb, text_emb], dim=1)

        enc = self.gpt(emb)

        # Compute loss
        text_logits = self.final_norm(enc[:, self.MAX_MEL_FRAMES:])
        text_logits = self.text_head(text_logits)
        text_logits = text_logits.permute(0,2,1)
        loss_text = F.cross_entropy(text_logits[:,:,:-1], text_targets[:,1:].long())

        return loss_text.mean()

    def inference_beam_topk(self, mel):
        def topk_sampler(distribution, k):
            return torch.topk(distribution, k=k, dim=-1)
        return self.inference_beam(mel, topk_sampler)

    def inference_beam_sampled(self, mel):
        def multinomial_sampler(distribution, k):
            indices = torch.multinomial(distribution, num_samples=k, replacement=False)
            values = torch.gather(distribution, dim=1, index=indices)
            class container:
                def __init__(self, i, v):
                    self.indices = i
                    self.values = v
            return container(indices, values)
        return self.inference_beam(mel, multinomial_sampler)

    def inference_beam(self, mel_inputs, sampler_fn):
        beam_width = 16
        temperature = .8

        b, _, s = mel_inputs.shape
        assert b == 1  # Beam search only works on batches of one.
        mel_emb = self.mel_encoder(mel_inputs)
        mel_emb = F.pad(mel_emb, (0, self.MAX_MEL_FRAMES-mel_emb.shape[-1]))
        mel_emb = mel_emb.permute(0,2,1).contiguous()
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_emb.device))

        text_seq = torch.full((b,1), fill_value=self.NUMBER_SYMBOLS, device=mel_emb.device)
        probabilities = torch.ones((b,), device=mel_emb.device)
        while len(text_seq) < self.max_mel_frames:
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


@register_model
def register_gpt_asr(opt_net, opt):
    return GptAsr(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    gpt = GptAsr()
    l = gpt(torch.randn(2,80,800),
               torch.randint(high=len(symbols), size=(2,180)))
    print(l.shape)

    #o = gpt.infer(torch.randint(high=24, size=(2,60)))
    #print(o.shape)


