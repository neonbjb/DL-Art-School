import functools
import math
import random
from functools import partial

import torch
import torch.nn as nn
from x_transformers.x_transformers import groupby_prefix_and_trim, FixedPositionalEmbedding, default, RotaryEmbedding, \
    DEFAULT_DIM_HEAD, RelativePositionBias, LearnedAlibiPositionalBias, AlibiPositionalBias, ScaleNorm, RMSNorm, \
    exists, Attention, FeedForward, Scale, ShiftTokens, GRUGating, Residual, cast_tuple, equals, LayerIntermediates, \
    AttentionLayers, not_equals


class TimeIntegrationBlock(nn.Module):
    def __init__(self, time_emb_dim, dim, normalizer):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_emb_dim,
                2 * dim
            ),
        )
        self.normalizer = normalizer

    def forward(self, x, time_emb):
        emb_out = self.emb_layers(time_emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        x = self.normalizer(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbeddingAttentionLayers(AttentionLayers):
    """
    Modification of x-transformers.AttentionLayers that performs timestep embeddings and layerdrop.
    """
    def __init__(
        self,
        dim,
        timestep_dim,
        depth,
        heads = 8,
        causal = False,
        cross_attend = False,
        only_cross = False,
        use_scalenorm = False,
        use_rmsnorm = False,
        use_rezero = False,
        alibi_pos_bias = False,
        alibi_num_heads = None,
        alibi_learned = False,
        rel_pos_bias = False,
        rel_pos_num_buckets = 32,
        rel_pos_max_distance = 128,
        position_infused_attn = False,
        rotary_pos_emb = False,
        rotary_emb_dim = None,
        custom_layers = None,
        sandwich_coef = None,
        par_ratio = None,
        residual_attn = False,
        cross_residual_attn = False,
        macaron = False,
        gate_residual = False,
        scale_residual = False,
        shift_tokens = 0,
        use_qk_norm_attn = False,
        qk_norm_attn_seq_len = None,
        zero_init_branch_output = False,
        layerdrop_percent = .1,
        **kwargs
    ):
        super().__init__(dim, depth)
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim('attn_', kwargs)

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])
        self.layerdrop_percent = layerdrop_percent

        self.has_pos_emb = position_infused_attn or rel_pos_bias or rotary_pos_emb
        self.pia_pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else None

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim) if rotary_pos_emb else None

        assert not (alibi_pos_bias and rel_pos_bias), 'you can only choose Alibi positional bias or T5 relative positional bias, not both'
        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'

        if rel_pos_bias:
            self.rel_pos = RelativePositionBias(scale = dim_head ** 0.5, causal = causal, heads = heads, num_buckets = rel_pos_num_buckets, max_distance = rel_pos_max_distance)
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, 'number of ALiBi heads must be less than the total number of heads'
            alibi_pos_klass = LearnedAlibiPositionalBias if alibi_learned or not causal else AlibiPositionalBias
            self.rel_pos = alibi_pos_klass(heads = alibi_num_heads, bidirectional = not causal)
        else:
            self.rel_pos = None

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        self.cross_attend = cross_attend

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)

        norm_fn = nn.Identity if use_rezero else norm_fn
        branch_fn = Rezero if use_rezero else None

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        # qk normalization
        if use_qk_norm_attn:
            attn_scale_init_value = -math.log(math.log2(qk_norm_attn_seq_len ** 2 - qk_norm_attn_seq_len)) if exists(qk_norm_attn_seq_len) else None
            attn_kwargs = {**attn_kwargs, 'qk_norm': True, 'scale_init_value': attn_scale_init_value}

        # zero init
        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, 'zero_init_output':  True}
            ff_kwargs = {**ff_kwargs, 'zero_init_output':  True}

        # calculate layer block order
        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn  = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_layer_types = len(set(self.layer_types))
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        # calculate token shifting
        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # iterate and construct layers
        for ind, (layer_type, layer_shift_tokens) in enumerate(zip(self.layer_types, shift_tokens)):
            if layer_type == 'a':
                layer = Attention(dim, heads = heads, causal = causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads = heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            if exists(branch_fn):
                layer = branch_fn(layer)

            residual_fn = GRUGating if gate_residual else Residual
            residual = residual_fn(dim, scale_residual = scale_residual)

            layer_uses_qk_norm = use_qk_norm_attn and layer_type in ('a', 'c')

            pre_branch_norm = TimeIntegrationBlock(timestep_dim, dim, norm_fn())
            post_branch_norm = norm_fn() if layer_uses_qk_norm else None
            post_main_norm = None  # Always do prenorm for timestep integration.

            norms = nn.ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm
            ])

            self.layers.append(nn.ModuleList([
                norms,
                layer,
                residual
            ]))

    def forward(
        self,
        x,
        time_emb = None,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        mems = None,
        return_hiddens = False
    ):
        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross_attend is set to True'
        assert time_emb is not None, 'must specify a timestep embedding.'

        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(list(map(lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1], mems)))
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length, x.device)

        unused_params = []
        to_drop = 0
        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            if layer_type == 'a':
                # Do layer drop where applicable. Do not drop first layer. When doing layer-drop, drop all of the joined layers (e.g. attention + context + feedforward)
                if self.training and self.layerdrop_percent > 0 and ind != 0 and random.random() < self.layerdrop_percent:
                    to_drop = self.num_layer_types

                hiddens.append(x)
                layer_mem = mems.pop(0) if mems else None

            if to_drop > 0:
                to_drop -= 1
                # Record the unused parameters so they can be used in null-operations later to not trigger DDP.
                unused_params.extend(list(block.parameters()))
                unused_params.extend(list(residual_fn.parameters()))
                unused_params.extend(list(norm.parameters()))
                continue

            residual = x

            pre_branch_norm, post_branch_norm, post_main_norm = norm

            x = pre_branch_norm(x, time_emb)

            if layer_type == 'a':
                out, inter = block(x, mask = mask, attn_mask = attn_mask, sinusoidal_emb = self.pia_pos_emb, rel_pos = self.rel_pos, rotary_pos_emb = rotary_pos_emb, prev_attn = prev_attn, mem = layer_mem)
            elif layer_type == 'c':
                out, inter = block(x, context, mask = mask, context_mask = context_mask, prev_attn = prev_cross_attn)
            elif layer_type == 'f':
                out = block(x)

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, residual)

            if layer_type in ('a', 'c'):
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        x = x + extraneous_addition * 0

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens = hiddens,
                attn_intermediates = intermediates
            )

            return x, intermediates

        return x