import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from trainer.inject import Injector
from trainer.networks import register_model
from utils.util import checkpoint


def create_injector(opt, env):
    type = opt['type']
    if type == 'igpt_resolve':
        return ResolveInjector(opt, env)
    return None


class ResolveInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.gen = opt['generator']
        self.samples = opt['num_samples']
        self.temperature = opt['temperature']

    def forward(self, state):
        gen = self.env['generators'][self.opt['generator']].module
        img = state[self.input]
        b, c, h, w = img.shape
        qimg = gen.quantize(img)
        s, b = qimg.shape
        qimg = qimg[:s//2, :]
        output = qimg.repeat(1, self.samples)

        pad = torch.zeros(1, self.samples, dtype=torch.long).cuda()  # to pad prev output
        with torch.no_grad():
            for _ in range(s//2):
                logits, _ = gen(torch.cat((output, pad), dim=0), already_quantized=True)
                logits = logits[-1, :, :] / self.temperature
                probs = F.softmax(logits, dim=-1)
                pred = torch.multinomial(probs, num_samples=1).transpose(1, 0)
                output = torch.cat((output, pred), dim=0)
        output = gen.unquantize(output.reshape(h, w, -1))
        return {self.output: output.permute(2,3,0,1).contiguous()}


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class iGPT2(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers, num_positions, num_vocab, centroids_file
    ):
        super().__init__()

        self.centroids = nn.Parameter(
            torch.from_numpy(np.load(centroids_file)), requires_grad=False
        )
        self.embed_dim = embed_dim

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.token_embeddings = nn.Embedding(num_vocab, embed_dim)
        self.position_embeddings = nn.Embedding(num_positions, embed_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vocab, bias=False)
        self.clf_head = nn.Linear(embed_dim, 10)  # Fixed num_classes, this is not a classifier.

    def squared_euclidean_distance(self, a, b):
        b = torch.transpose(b, 0, 1)
        a2 = torch.sum(torch.square(a), dim=1, keepdims=True)
        b2 = torch.sum(torch.square(b), dim=0, keepdims=True)
        ab = torch.matmul(a, b)
        d = a2 - 2 * ab + b2
        return d

    def quantize(self, x):
        b, c, h, w = x.shape
        # [B, C, H, W] => [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, c)  # flatten to pixels
        d = self.squared_euclidean_distance(x, self.centroids)
        x = torch.argmin(d, 1)
        x = x.view(b, h, w)

        # Reshape output to [seq_len, batch].
        x = x.view(x.shape[0], -1)  # flatten images into sequences
        x = x.transpose(0, 1).contiguous()  # to shape [seq len, batch]
        return x

    def unquantize(self, x):
        return self.centroids[x]

    def forward(self, x, already_quantized=False):
        """
        Expect input as shape [b, c, h, w]
        """

        if not already_quantized:
            x = self.quantize(x)
        length, batch = x.shape

        h = self.token_embeddings(x)

        # prepend sos token
        sos = torch.ones(1, batch, self.embed_dim, device=x.device) * self.sos
        h = torch.cat([sos, h[:-1, :, :]], axis=0)

        # add positional embeddings
        positions = torch.arange(length, device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)

        # transformer
        for layer in self.layers:
            h = checkpoint(layer, h)

        h = self.ln_f(h)

        logits = self.head(h)

        return logits, x


@register_model
def register_igpt2(opt_net, opt):
    return iGPT2(opt_net['embed_dim'], opt_net['num_heads'], opt_net['num_layers'], opt_net['num_pixels'] ** 2,
          opt_net['num_vocab'], centroids_file=opt_net['centroids_file'])
