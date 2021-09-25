import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from models.switched_conv.switched_conv_hard_routing import SwitchNorm
from utils.weight_scheduler import LinearDecayWeightScheduler


class GumbelQuantizer(nn.Module):
    def __init__(self, inp_dim, codebook_dim, num_tokens, straight_through=False):
        super().__init__()
        self.to_logits = nn.Conv1d(inp_dim, num_tokens, 1)
        self.codebook = nn.Embedding(num_tokens, codebook_dim)
        self.straight_through = straight_through
        self.temperature_scheduler = LinearDecayWeightScheduler(10, 5000, .9, 2000)
        self.step = 0
        self.norm = SwitchNorm(num_tokens)

    def get_temperature(self, step):
        self.step = step  # VERY POOR DESIGN. WHEN WILL HE EVER LEARN???
        return self.temperature_scheduler.get_weight_for_step(step)

    def embed_code(self, codes):
        return self.codebook(codes)

    def gumbel_softmax(self, logits, tau, dim, hard):
        gumbels = torch.rand_like(logits)
        gumbels = -torch.log(-torch.log(gumbels + 1e-8) + 1e-8)
        logits = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = F.softmax(logits, dim=dim)

        if hard:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    def forward(self, h):
        h = h.permute(0,2,1)
        logits = self.to_logits(h)
        logits = self.gumbel_softmax(logits, tau=self.temperature_scheduler.get_weight_for_step(self.step), dim=1, hard=self.straight_through)
        logits = self.norm(logits)
        codes = logits.argmax(dim=1).flatten(1)
        sampled = einsum('b n l, n d -> b d l', logits, self.codebook.weight)
        return sampled.permute(0,2,1), 0, codes

if __name__ == '__main__':
    from models.diffusion.diffusion_dvae import DiscreteDecoder
    j =  torch.randn(8,40,1024)
    m = GumbelQuantizer(1024, 1024, 4096)
    m2 = DiscreteDecoder(1024, (512, 256), 2)
    l=m2(m(j)[0].permute(0,2,1))
    mean = 0
    for ls in l:
        mean = mean + ls.mean()
    mean.backward()