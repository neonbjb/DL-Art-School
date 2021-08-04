import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import munchify
from torch import LongTensor
from tqdm import tqdm

from models.arch_util import ConvGnSilu
from models.gpt_voice.pixelshuffle_1d import PixelUnshuffle1D, PixelShuffle1D
from models.tacotron2 import hparams
from models.tacotron2.taco_utils import get_mask_from_lengths
from models.tacotron2.tacotron2 import Postnet
from models.tacotron2.text import symbols
from models.gpt_voice.min_gpt import GPT, GPTConfig
from trainer.networks import register_model


class GptTts(nn.Module):
    NUMBER_SYMBOLS = len(symbols)+3
    TEXT_START_TOKEN = NUMBER_SYMBOLS-3
    TEXT_STOP_TOKEN = NUMBER_SYMBOLS-2
    TEXT_PAD_TOKEN = NUMBER_SYMBOLS-1
    MEL_DICTIONARY_SIZE = 512+3
    MEL_START_TOKEN = MEL_DICTIONARY_SIZE-3
    MEL_STOP_TOKEN = MEL_DICTIONARY_SIZE-2
    MEL_PAD_TOKEN = MEL_DICTIONARY_SIZE-1

    def __init__(self):
        super().__init__()
        model_dim = 512
        max_symbols_per_phrase = 200
        max_mel_frames = 900 * 3 // 8  #  The VQVAE outputs 3/8 of the input mel as tokens.
        mel_dim=80

        self.model_dim = model_dim
        self.max_mel_frames = max_mel_frames
        self.text_embedding = nn.Embedding(self.NUMBER_SYMBOLS, model_dim)
        self.mel_embedding = nn.Embedding(self.MEL_DICTIONARY_SIZE, model_dim)
        # *_tags are additively applied to
        self.text_pos_embedding = nn.Embedding(max_symbols_per_phrase, model_dim)
        self.mel_pos_embedding = nn.Embedding(max_mel_frames, model_dim)
        self.gpt = GPT(GPTConfig(1+max_symbols_per_phrase+max_mel_frames, n_embd=model_dim, n_head=8), do_pos_emb=False)

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.NUMBER_SYMBOLS)
        self.mel_head = nn.Linear(model_dim, self.MEL_DICTIONARY_SIZE)

    def forward(self, text_inputs, text_lengths, mel_targets, output_lengths):
        text_emb = self.text_embedding(text_inputs)
        text_emb = text_emb + self.text_pos_embedding(torch.arange(text_inputs.shape[1], device=text_inputs.device))
        mel_emb = self.mel_embedding(mel_targets)
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_targets.shape[1], device=mel_targets.device))
        emb = torch.cat([text_emb, mel_emb], dim=1)
        enc = self.gpt(emb)

        # Compute logits for text and mel heads
        text_logits = self.final_norm(enc[:, :text_emb.shape[1]])
        text_logits = self.text_head(text_logits)
        mel_logits = self.final_norm(enc[:, text_emb.shape[1]:])
        mel_logits = self.mel_head(mel_logits)

        # Compute loss
        loss_text = F.cross_entropy(text_logits.permute(0,2,1)[:,:,1:], text_inputs[:,1:], reduction='none')
        loss_mel = F.cross_entropy(mel_logits.permute(0,2,1)[:,:,1:], mel_targets[:,1:], reduction='none')

        # Apply a reduction factor across MEL_PAD and TEXT_PAD tokens.
        pad_loss_reduction_factor = .01
        text_pad_mask = ~get_mask_from_lengths(text_lengths, text_inputs.shape[1])
        mel_pad_mask = ~get_mask_from_lengths(output_lengths, mel_targets.shape[1])
        loss_text = loss_text * torch.ones_like(loss_text).masked_fill_(text_pad_mask[:,1:], pad_loss_reduction_factor)
        loss_mel = loss_mel * torch.ones_like(loss_mel).masked_fill_(mel_pad_mask[:,1:], pad_loss_reduction_factor)

        # Fix up mel_logits so it can go into a VAE decoder as well.
        mel_codes = torch.argmax(F.softmax(mel_logits, dim=-1), dim=-1)
        mel_codes = mel_codes[:,1:-1]  # Strip off first and last tokens (START+STOP were added by the dataloader)
        mel_codes = mel_codes * torch.ones_like(mel_codes).masked_fill_(mel_pad_mask[:,1:-1], 0)
        extra_mask = mel_codes < self.MEL_DICTIONARY_SIZE-3  # The VAE doesn't know about START/STOP/PAD
        mel_codes = mel_codes * extra_mask

        return loss_text.mean(), loss_mel.mean(), mel_codes

    def inference(self, text_inputs):
        text_emb = self.text_embedding(text_inputs)
        text_emb = text_emb + self.text_pos_embedding(torch.arange(text_inputs.shape[1], device=text_inputs.device))

        mel_seq = [self.MEL_START_TOKEN, 0]
        while mel_seq[-1] != self.MEL_STOP_TOKEN and len(mel_seq) < self.max_mel_frames:
            mel_emb = self.mel_embedding(LongTensor(mel_seq, device=text_inputs.device))
            mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_seq.shape[1], device=mel_seq.device))
            emb = torch.cat([text_emb, mel_emb], dim=1)
            enc = self.gpt(emb)
            mel_logits = self.final_norm(enc[:, text_emb.shape[1]:])
            mel_logits = self.mel_head(mel_logits)
            mel_codes = torch.argmax(F.softmax(mel_logits, dim=-1), dim=-1)
            mel_seq[-1] = mel_codes[-1]
            mel_seq.append(0)

        if len(mel_seq) >= self.max_mel_frames:
            print("Warning! Encountered frame limit before a stop token. Output is likely wrong.")

        return mel_seq[:-1]


@register_model
def register_gpt_tts(opt_net, opt):
    return GptTts()


if __name__ == '__main__':
    gpt = GptTts()
    l1, l2, i = gpt(torch.randint(high=24, size=(2,60)),
               torch.tensor([55,58]),
               torch.randint(high=512, size=(2,310)),
               torch.tensor([300,305]))
    print(i.shape)

    #o = gpt.infer(torch.randint(high=24, size=(2,60)))
    #print(o.shape)


