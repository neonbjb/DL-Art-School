import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from trainer.networks import register_model
from utils.util import opt_get


class Mel2VecCodesGpt(nn.Module):
    def __init__(self, dim, layers, num_groups=8, num_vectors=8):
        super().__init__()

        self.num_groups = num_groups

        self.config = GPT2Config(vocab_size=1, n_positions=8192, n_embd=dim, n_layer=layers, n_head=dim//64,
                                 n_inner=dim*2)
        self.gpt = GPT2Model(self.config)
        del self.gpt.wte  # Unused, we'll do our own embeddings.
        self.embeddings = nn.ModuleList([nn.Embedding(num_vectors, dim//num_groups) for _ in range(num_groups)])
        self.heads = nn.ModuleList([nn.Linear(dim, num_vectors) for _ in range(num_groups)])

    def forward(self, codes):
        assert codes.shape[-1] == self.num_groups

        inputs = codes[:, :-1]
        targets = codes[:, 1:]

        h = [embedding(inputs[:, :, i]) for i, embedding in enumerate(self.embeddings)]
        h = torch.cat(h, dim=-1)
        h = self.gpt(inputs_embeds=h, return_dict=True).last_hidden_state

        losses = 0
        for i, head in enumerate(self.heads):
            logits = head(h).permute(0,2,1)
            loss = F.cross_entropy(logits, targets[:,:,i])
            losses = losses + loss

        return losses / self.num_groups


@register_model
def register_music_gpt(opt_net, opt):
    return Mel2VecCodesGpt(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    model = Mel2VecCodesGpt(512, 8)
    codes = torch.randint(0,8, (2,300,8))
    model(codes)