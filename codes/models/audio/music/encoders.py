import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2Model

from models.arch_util import AttentionBlock, ResBlock
from models.audio.tts.lucidrains_dvae import DiscreteVAE
from trainer.networks import register_model
from utils.util import opt_get, ceil_multiple, print_network


class ResEncoder16x(nn.Module):
    def __init__(self,
                 spec_dim,
                 hidden_dim,
                 embedding_dim,
                 checkpointing_enabled=True,
                 ):
        super().__init__()
        attn = []
        def edim(m):
            dd = min(spec_dim + m * 128, hidden_dim)
            return ceil_multiple(dd, 8)
        self.downsampler = nn.Sequential(
            ResBlock(spec_dim, out_channels=edim(2), use_conv=True, dims=1, down=True, checkpointing_enabled=checkpointing_enabled),
            ResBlock(edim(2), out_channels=edim(3), use_conv=True, dims=1, down=True, checkpointing_enabled=checkpointing_enabled),
            ResBlock(edim(3), out_channels=edim(3), use_conv=True, dims=1, checkpointing_enabled=checkpointing_enabled),            
            ResBlock(edim(3), out_channels=edim(4), use_conv=True, dims=1, down=True, checkpointing_enabled=checkpointing_enabled),
            ResBlock(edim(4), out_channels=edim(4), use_conv=True, dims=1, checkpointing_enabled=checkpointing_enabled),
            ResBlock(edim(4), out_channels=hidden_dim, use_conv=True, dims=1, down=True, checkpointing_enabled=checkpointing_enabled))
        self.encoder = nn.Sequential(
            ResBlock(hidden_dim, out_channels=hidden_dim, use_conv=True, dims=1, checkpointing_enabled=checkpointing_enabled),
            ResBlock(hidden_dim, out_channels=hidden_dim, use_conv=True, dims=1, checkpointing_enabled=checkpointing_enabled),
            ResBlock(hidden_dim, out_channels=hidden_dim, use_conv=True, dims=1, checkpointing_enabled=checkpointing_enabled),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, embedding_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.downsampler(x)
        h = self.encoder(h)
        return h
