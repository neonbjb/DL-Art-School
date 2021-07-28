import torch
import torch.nn as nn
import torch.nn.functional as F

from models.arch_util import ConvGnSilu
from models.tacotron2.taco_utils import get_mask_from_lengths
from models.tacotron2.text import symbols
from models.gpt_voice.min_gpt import GPT, GPTConfig
from trainer.networks import register_model


class GptTts(nn.Module):
    def __init__(self):
        super().__init__()
        number_symbols = len(symbols)
        model_dim = 512
        max_symbols_per_phrase = 200
        max_mel_frames = 900
        mel_dim=80

        self.text_embedding = nn.Embedding(number_symbols, model_dim)
        self.mel_encoder = nn.Sequential(ConvGnSilu(mel_dim, model_dim//2, kernel_size=3, convnd=nn.Conv1d),
                                         ConvGnSilu(model_dim//2, model_dim, kernel_size=3, stride=2, convnd=nn.Conv1d))
        self.text_tags = nn.Parameter(torch.randn(1, 1, model_dim)/256.0)
        self.audio_tags = nn.Parameter(torch.randn(1, 1, model_dim)/256.0)
        self.gpt = GPT(GPTConfig(max_symbols_per_phrase+max_mel_frames//2, n_embd=model_dim, n_head=8))

        self.gate_head = nn.Sequential(ConvGnSilu(model_dim, model_dim, kernel_size=5, convnd=nn.Conv1d),
                                       nn.Upsample(scale_factor=2, mode='nearest'),
                                       ConvGnSilu(model_dim, model_dim//2, kernel_size=5, convnd=nn.Conv1d),
                                       nn.Conv1d(model_dim//2, 1, kernel_size=1))
        self.mel_head = nn.Sequential(ConvGnSilu(model_dim, model_dim, kernel_size=5, convnd=nn.Conv1d),
                                      nn.Upsample(scale_factor=2, mode='nearest'),
                                      ConvGnSilu(model_dim, model_dim//2, kernel_size=5, convnd=nn.Conv1d),
                                      ConvGnSilu(model_dim//2, model_dim//2, kernel_size=5, convnd=nn.Conv1d),
                                      ConvGnSilu(model_dim//2, mel_dim, kernel_size=1, activation=False, norm=False, convnd=nn.Conv1d))

    def forward(self, text_inputs, mel_targets, output_lengths):
        # Pad mel_targets to be a multiple of 2
        padded = mel_targets.shape[-1] % 2 != 0
        if padded:
            mel_targets = F.pad(mel_targets, (0,1))

        text_emb = self.text_embedding(text_inputs)
        text_emb = text_emb + self.text_tags
        mel_emb = self.mel_encoder(mel_targets).permute(0,2,1)
        mel_emb = mel_emb + self.audio_tags
        emb = torch.cat([text_emb, mel_emb], dim=1)
        enc = self.gpt(emb)
        mel_portion = enc[:, text_emb.shape[1]:].permute(0,2,1)
        gates = self.gate_head(mel_portion).squeeze(1)
        mel_pred = self.mel_head(mel_portion)

        # Mask portions of output which we don't need to predict.
        mask = ~get_mask_from_lengths(output_lengths, mel_pred.shape[-1])
        mask = mask.unsqueeze(1).repeat(1, mel_pred.shape[1], 1)
        mel_pred.data.masked_fill_(mask, 0)
        gates.data.masked_fill_(mask[:, 0, :], 1e3)

        if padded:
            mel_pred = mel_pred[:, :, :-1]
            gates = gates[:, :-1]
        return mel_pred, gates


@register_model
def register_gpt_tts(opt_net, opt):
    return GptTts()


if __name__ == '__main__':
    gpt = GptTts()
    m, g = gpt(torch.randint(high=24, size=(2,60)),
               torch.randn(2,80,747),
               torch.tensor([600,747]))
    print(m.shape)
    print(g.shape)