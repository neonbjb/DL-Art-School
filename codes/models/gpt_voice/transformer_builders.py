"""
A list of functions that map a unified set of arguments to a fully built transformer. Also includes some testing
utilities for measuring parameter count, FLOPS, and general performance of each type.

Every function contains the following arguments:

        layers: Net number of layers in the transformer.
        model_dim: Hidden dimensionality of the model.
        heads: Number of attention heads.
        num_tokens: Number of possible tokens in the transformer's dictionary. Do not use this in future releases.
        max_seq_len: Maximum sequence length to attend to.
        checkpointing: Whether or not the underlying implementation should support gradient checkpointing.
"""
import functools
import torch


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


def build_hf_gpt_transformer(layers, model_dim, heads, num_tokens, max_seq_len, checkpointing):
    """
    GPT-2 implemented by the HuggingFace library.
    """
    from transformers import GPT2Config, GPT2Model
    gpt_config = GPT2Config(vocab_size=num_tokens,
                                 n_positions=max_seq_len,
                                 n_ctx=max_seq_len,
                                 n_embd=model_dim,
                                 n_layer=layers,
                                 n_head=heads,
                                 gradient_checkpointing=checkpointing,
                                 use_cache=not checkpointing)
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    return gpt


def build_lr_performer(layers, model_dim, heads, num_tokens, max_seq_len, checkpointing):
    """
    lucidrains Performer implementation, https://github.com/lucidrains/performer-pytorch
    """
    pass


def build_lr_reformer(layers, model_dim, heads, num_tokens, max_seq_len, checkpointing):
    """
    lucidrains Reformer implementation, https://github.com/lucidrains/reformer-pytorch
    """
    pass


def build_lr_xformer(layers, model_dim, heads, num_tokens, max_seq_len, checkpointing):
    """
    lucidrains x-transformer implementation, https://github.com/lucidrains/x-transformers
    """
    pass


def test_all_performance(**kwargs):
    transformer_builders = [build_hf_gpt_transformer, build_lr_performer, build_lr_reformer, build_lr_xformer]
    for builder in transformer_builders:
        model = builder(**kwargs)


if __name__ == '__main__':
    test_all_performance(12, 512, 8, 8192, 1000, False)