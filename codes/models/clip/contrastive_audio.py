from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from models.arch_util import AttentionBlock
from models.lucidrains.x_transformers import ContinuousTransformerWrapper, Encoder
from trainer.networks import register_model
from utils.util import opt_get, checkpoint


def exists(val):
    return val is not None


def masked_mean(t, mask):
    t = t.masked_fill(~mask, 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class CollapsingTransformer(nn.Module):
    def __init__(self, model_dim, output_dims, heads, dropout, depth, mask_percentage=0, **encoder_kwargs):
        super().__init__()
        self.transformer = ContinuousTransformerWrapper(
            max_seq_len=-1,
            use_pos_emb=False,
            attn_layers=Encoder(
                dim=model_dim,
                depth=depth,
                heads=heads,
                ff_dropout=dropout,
                ff_mult=1,
                attn_dropout=dropout,
                use_rmsnorm=True,
                ff_glu=True,
                rotary_pos_emb=True,
                **encoder_kwargs,
            ))
        self.pre_combiner = nn.Sequential(nn.Conv1d(model_dim, output_dims, 1),
                                          AttentionBlock(output_dims, num_heads=heads, do_checkpoint=False),
                                          nn.Conv1d(output_dims, output_dims, 1))
        self.mask_percentage = mask_percentage

    def forward(self, x, **transformer_kwargs):
        h = self.transformer(x, **transformer_kwargs)
        h = h.permute(0,2,1)
        h = checkpoint(self.pre_combiner, h).permute(0,2,1)
        if self.training:
            mask = torch.rand_like(h.float()) > self.mask_percentage
        else:
            mask = torch.ones_like(h.float()).bool()
        return masked_mean(h, mask)


class ConvFormatEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(*args, **kwargs)

    def forward(self, x):
        y = self.emb(x)
        return y.permute(0,2,1)


class ContrastiveAudio(nn.Module):
    def __init__(
            self,
            model_dim=512,
            transformer_heads=8,
            dropout=.1,
            encoder_depth=8,
            mel_channels=80,
            latent_multiplier=1,
            mask_percent=.15,
    ):
        super().__init__()
        latent_dim = latent_multiplier*model_dim
        self.temperature = nn.Parameter(torch.tensor(1.))

        self.emb = nn.Sequential(nn.Conv1d(mel_channels, model_dim // 2, kernel_size=5, stride=2, padding=2),
                                 nn.Conv1d(model_dim//2, model_dim, kernel_size=3, stride=2, padding=1))
        self.transformer = CollapsingTransformer(model_dim, model_dim, transformer_heads, dropout, encoder_depth, mask_percent)
        self.to_latent = nn.Linear(latent_dim, latent_dim, bias=False)
        self.to_latent2 = nn.Linear(latent_dim, latent_dim, bias=False)

        self.to_latent2.weight.data = self.to_latent.weight.data
        self.to_latent2.weight.DO_NOT_TRAIN = True
        self.to_latent2.requires_grad = False

    def get_grad_norm_parameter_groups(self):
        return {
            'emb': list(self.emb.parameters()),
            'xform': list(self.transformer.parameters()),
        }

    def update_for_step(self, step, __):
        self.to_latent2.weight.data = self.to_latent2.weight.data * .99 + self.to_latent.weight.data * .01

    def project(self, mel):
        h1 = self.emb(mel).permute(0, 2, 1)
        h1 = self.transformer(h1)
        h1 = self.to_latent(h1)
        return h1

    def forward(
            self,
            mel_input1,
            mel_input2
    ):
        if len(mel_input2.shape) == 4:
            mel_input2 = mel_input2[:, 0]
        if self.training:
            # Mask out big chunks of separate frequency bands for each clip.
            b, c, _ = mel_input1.shape
            mask = torch.rand(b,c,1, device=mel_input1.device) > .3
            mel_input1 = mask * mel_input1 * (1-random()*.5)
            mask = torch.rand(b,c,1, device=mel_input2.device) > .3
            mel_input2 = mask * mel_input2 * (1-random()*.5)

        h1 = self.emb(mel_input1).permute(0, 2, 1)
        h1 = self.transformer(h1)
        h1 = self.to_latent(h1)

        h2 = self.emb(mel_input2).permute(0, 2, 1)
        h2 = self.transformer(h2)
        h2 = self.to_latent2(h2).detach()

        loss = info_nce(h1, h2)
        return loss


@register_model
def register_contrastive_audio(opt_net, opt):
    return ContrastiveAudio(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    clvp = ContrastiveAudio()
    clvp(torch.randn(2,80,100),
         torch.randn(2,80,95),
         return_loss=True)
    v = torch.randn(2,512)
    print(info_nce(v,v))