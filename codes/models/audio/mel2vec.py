import copy
import functools
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from transformers.deepspeed import is_deepspeed_zero3_enabled

from trainer.networks import register_model
from utils.util import checkpoint


class Mel2Vec2FeatureProjection(nn.Module):
    def __init__(self, inner_dim, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(inner_dim, eps=1e-5)
        self.projection = nn.Linear(inner_dim, inner_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Wav2Vec2
class Wav2Vec2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class Wav2Vec2FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(dropout)

        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = F.gelu

        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=hidden_size,
            num_heads=hidden_size//64,
            dropout=dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.feed_forward = Wav2Vec2FeedForward(hidden_size, hidden_size*2, dropout)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Wav2Vec2SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


from torch.nn.utils.weight_norm import WeightNorm
def __deepcopy__(self, memo):
    # save and delete all weightnorm weights on self
    weights = {}
    for hook in self._forward_pre_hooks.values():
        if isinstance(hook, WeightNorm):
            weights[hook.name] = getattr(self, hook.name)
            delattr(self, hook.name)
    # remove this deepcopy method, restoring the object's original one if necessary
    __deepcopy__ = self.__deepcopy__
    if self.orig_deepcopy:
        self.__deepcopy__ = self.orig_deepcopy
    else:
        del self.__deepcopy__
    # actually do the copy
    result = copy.deepcopy(self)
    # restore weights and method on self
    for name, value in weights.items():
        setattr(self, name, value)
    self.__deepcopy__ = __deepcopy__
    return result


class Wav2Vec2PositionalConvEmbedding(nn.Module):
    def __init__(self, hidden_size, num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=num_conv_pos_embeddings,
            padding=num_conv_pos_embeddings // 2,
            groups=num_conv_pos_embedding_groups,
        )
        # Fix weightnorm deepcopy; see: https://github.com/pytorch/pytorch/issues/28594
        self.conv.orig_deepcopy = getattr(Wav2Vec2PositionalConvEmbedding, '__deepcopy__', None)
        self.conv.__deepcopy__ = __deepcopy__.__get__(self.conv, self.conv.__class__)

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

        self.padding = Wav2Vec2SamePadLayer(num_conv_pos_embeddings)
        self.activation = F.gelu

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Wav2Vec2Encoder(nn.Module):
    def __init__(self, hidden_size, dropout, num_layers, layerdrop):
        super().__init__()
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([Wav2Vec2EncoderLayer(hidden_size, dropout) for _ in range(num_layers)])
        self.layerdrop = layerdrop

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states[~attention_mask] = 0.0

            # extend attention_mask
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = True if self.training and (dropout_probability < self.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                layer_fn = functools.partial(layer, attention_mask=attention_mask)
                layer_outputs = checkpoint(layer_fn, hidden_states)
                hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class Mel2Vec(nn.Module):
    def __init__(self,
                 mel_input_channels=256,
                 inner_dim=1024,
                 layers=24,
                 dropout=.1,
                 layerdrop=0,
                 mask_time_prob=.65,
                 mask_time_length=10,
                 ):
        super().__init__()
        self.input_blocks = nn.Sequential(nn.Conv1d(mel_input_channels, inner_dim//2, kernel_size=5, padding=2, stride=2),
                                          nn.GroupNorm(num_groups=8, num_channels=inner_dim//2, affine=True),
                                          nn.SiLU(),
                                          nn.Conv1d(inner_dim//2, inner_dim,  kernel_size=3, padding=1, stride=2),
                                          nn.GroupNorm(num_groups=8, num_channels=inner_dim, affine=True),
                                          nn.SiLU(),
                                          )
        self.projector = Mel2Vec2FeatureProjection(inner_dim, dropout)
        self.masked_spec_embed = nn.Parameter(torch.rand(inner_dim,))
        self.encoder = Wav2Vec2Encoder(inner_dim, dropout, layers, layerdrop)
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.apply(self.init)

    def init(self, module):
        """Initialize the weights"""
        # gumbel softmax requires special init
        if isinstance(module, Wav2Vec2PositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, Mel2Vec2FeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def apply_masking(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.mask_time_prob,
                mask_length=self.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        return hidden_states

    def forward(self, mel, mask_time_indices=None, return_projections=False):
        proj = self.input_blocks(mel).permute(0,2,1)
        proj, _ = self.projector(proj)

        # Mask projections
        h = self.apply_masking(proj, mask_time_indices)
        h = self.encoder(h)

        if return_projections:
            return h, proj
        return h


class Wav2Vec2GumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See `[CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """

    def __init__(self, proj_dim=1024, codevector_dim=256, num_codevector_groups=2, num_codevectors_per_group=320):
        super().__init__()
        self.codevector_dim = codevector_dim
        self.num_groups = num_codevector_groups
        self.num_vars = num_codevectors_per_group
        self.num_codevectors = num_codevector_groups * num_codevectors_per_group

        if codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`codevector_dim {codevector_dim} must be divisible "
                f"by `num_codevector_groups` {num_codevector_groups} for concatenation"
            )

        # storage for codebook variables (codewords)
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, codevector_dim // self.num_groups)
        )
        self.weight_proj = nn.Linear(proj_dim, self.num_groups * self.num_vars)

        # can be decayed for training
        self.temperature = 2

        # Parameters init.
        self.weight_proj.weight.data.normal_(mean=0.0, std=1)
        self.weight_proj.bias.data.zero_()
        nn.init.uniform_(self.codevectors)

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)

        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity

    def forward(self, hidden_states, mask_time_indices=None):
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # sample code vector probs via gumbel in differentiateable way
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # compute perplexity
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # take argmax in non-differentiable way
            # comptute hard codevector distribution (one hot)
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # use probs to retrieve codevectors
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = (
            codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
            .sum(-2)
            .view(batch_size, sequence_length, -1)
        )

        return codevectors, perplexity


class ContrastiveTrainingWrapper(nn.Module):
    def __init__(self, inner_dim=1024, dropout=.1, mask_time_prob=.65, mask_time_length=4, num_negatives=100,
                 max_gumbel_temperature=2.0, min_gumbel_temperature=.5, gumbel_temperature_decay=.999995, **kwargs):
        super().__init__()
        self.m2v = Mel2Vec(inner_dim=inner_dim, dropout=dropout, mask_time_prob=mask_time_prob,
                           mask_time_length=mask_time_length, **kwargs)
        self.dropout_features = nn.Dropout(dropout)
        self.num_negatives = num_negatives
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.max_gumbel_temperature = max_gumbel_temperature
        self.min_gumbel_temperature = min_gumbel_temperature
        self.gumbel_temperature_decay = gumbel_temperature_decay

        self.quantizer = Wav2Vec2GumbelVectorQuantizer(inner_dim)

        # make sure that project_hid & project_q are initialized like normal linear layers
        self.project_hid = nn.Linear(inner_dim, self.quantizer.codevector_dim)
        self.project_q = nn.Linear(self.quantizer.codevector_dim, self.quantizer.codevector_dim)

    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: int = 0.1,
    ):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        target_features = torch.cat([target_features, negative_features], dim=0)

        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(
            target_features
        )

        # apply temperature
        logits = logits / temperature
        return logits

    def update_for_step(self, step, *args):
        self.quantizer.temperature = max(
                    self.max_gumbel_temperature * self.gumbel_temperature_decay**step,
                    self.min_gumbel_temperature,
                )

    def get_grad_norm_parameter_groups(self):
        if self.freeze_main_net:
            return {}
        groups = {
            'projector': list(self.m2v.input_blocks.parameters()) + list(self.m2v.projector.parameters()),
            'encoder': list(self.m2v.encoder.parameters()),
            'output_blocks': list(self.project_hid.parameters()) + list(self.project_q.parameters()),
        }
        return groups

    def forward(self, mel):
        mel = mel[:, :, :-1]  # The MEL computation always pads with 1, throwing off optimal tensor math.

        features_shape = (mel.shape[0], mel.shape[-1]//4)
        mask_time_indices = _compute_mask_indices(features_shape, self.mask_time_prob, self.mask_time_length)
        sampled_negative_indices = torch.tensor(_sample_negative_indices(features_shape, self.num_negatives, mask_time_indices=mask_time_indices), device=mel.device)
        mask_time_indices = torch.tensor(mask_time_indices, device=mel.device)

        outputs, proj = self.m2v(mel, mask_time_indices, return_projections=True)

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(outputs)

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(proj)

        quantized_features, codevector_perplexity = self.quantizer(
            extract_features, mask_time_indices=mask_time_indices
        )
        quantized_features = self.project_q(quantized_features)
        batch_size, sequence_length, hidden_size = quantized_features.shape

        # 3. sample K negatives (distractors) quantized states for contrastive loss
        # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
        # sample negative quantized vectors BTC => (BxT)C
        negative_quantized_features = quantized_features.view(-1, hidden_size)[
            sampled_negative_indices.long().view(-1)
        ]
        negative_quantized_features = negative_quantized_features.view(
            batch_size, sequence_length, -1, hidden_size
        ).permute(2, 0, 1, 3)

        # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
        # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
        logits = self.compute_contrastive_logits(
            quantized_features[None, :],
            negative_quantized_features,
            transformer_features,
            .1,
        )

        # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
        # its cosine similarity will be masked
        neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
        # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
        logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
        target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

        contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")
        # 7. compute diversity loss: \mathbf{L}_d
        num_codevectors = self.quantizer.num_codevectors
        diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time_indices.sum()

        return contrastive_loss, diversity_loss


@register_model
def register_mel2vec_pretraining(opt_net, opt):
    return ContrastiveTrainingWrapper(**opt_net['kwargs'])


@register_model
def register_mel2vec(opt_net, opt):
    return Mel2Vec(**opt_net['kwargs'])


if __name__ == '__main__':
    model = ContrastiveTrainingWrapper()
    mel = torch.randn((2,256,400))
    print(model(mel))