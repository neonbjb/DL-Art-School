import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


class GumbelQuantizer(nn.Module):
    def __init__(self, inp_dim, codebook_dim, num_tokens, straight_through=False, temperature=.9):
        super().__init__()
        self.to_logits = nn.Conv1d(inp_dim, num_tokens, 1)
        self.codebook = nn.Embedding(num_tokens, codebook_dim)
        self.straight_through = straight_through
        self.temperature = temperature

    def embed_code(self, codes):
        return self.codebook(codes)

    def forward(self, h):
        h = h.permute(0,2,1)
        logits = self.to_logits(h)
        logits = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=self.straight_through)
        codes = logits.argmax(dim=1).flatten(1)
        sampled = einsum('b n l, n d -> b d l', logits, self.codebook.weight)
        return sampled.permute(0,2,1), 0, codes