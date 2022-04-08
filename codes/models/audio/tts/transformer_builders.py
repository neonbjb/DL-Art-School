"""
A list of functions that map a unified set of arguments to a fully built transformer. Also includes some testing
utilities for measuring parameter count, FLOPS, and general performance of each type.

Every function contains the following arguments:

        layers: Net number of layers in the transformer.
        model_dim: Hidden dimensionality of the model.
        heads: Number of attention heads.
        max_mel_seq_len: Maximum mel sequence length to attend to.
        max_text_seq_len: Maximum text sequence length to attend to.
        checkpointing: Whether or not the underlying implementation should support gradient checkpointing.

Returns:
    (model, global_mel_pos_embedding, global_text_pos_embedding, local_mel_pos_embedding, local_text_pos_embedding)
    model: The transformer model
    global_mel_pos_embedding: A global embedding function (that takes the MEL sequence as input) which should be added on to the MEL embeddings.
    global_text_pos_embedding: The global embedding function for text tokens.
    local_mel_pos_embedding: A local embedding function which, if not None, should be concatenated with the local text position embeddings and fed to the transformer.
    local_text_pos_embedding: The local embedding function for text positions which will be None if local_mel_pos_embedding=None.

"""
import functools
import random
from time import time
import torch
import torch.nn as nn
from tqdm import tqdm


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02, relative=False):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            return self.emb(torch.arange(start, start+sl, device=x.device))
        else:
            return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


def build_hf_gpt_transformer(layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing):
    """
    GPT-2 implemented by the HuggingFace library.
    """
    from transformers import GPT2Config, GPT2Model
    gpt_config = GPT2Config(vocab_size=256,  # Unused.
                             n_positions=max_mel_seq_len+max_text_seq_len,
                             n_ctx=max_mel_seq_len+max_text_seq_len,
                             n_embd=model_dim,
                             n_layer=layers,
                             n_head=heads,
                             gradient_checkpointing=checkpointing,
                             use_cache=not checkpointing)
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte

    mel_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, model_dim) if max_mel_seq_len != -1 else functools.partial(null_position_embeddings, dim=model_dim)
    text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, model_dim) if max_mel_seq_len != -1 else functools.partial(null_position_embeddings, dim=model_dim)
    return gpt, mel_pos_emb, text_pos_emb, None, None


def build_lr_performer(layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing):
    """
    lucidrains Performer implementation, https://github.com/lucidrains/performer-pytorch
    """
    from models.lucidrains.performer.performer_pytorch import Performer
    model = Performer(dim=model_dim, depth=layers, heads=heads, dim_head=model_dim, causal=True)
    return model


def build_lr_reformer(layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing):
    """
    lucidrains Reformer implementation, https://github.com/lucidrains/reformer-pytorch
    """
    pass


def build_lr_xformer(layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing):
    """
    lucidrains x-transformer implementation, https://github.com/lucidrains/x-transformers
    """
    pass


def test_all_performance(**kwargs):
    transformer_builders = [#build_hf_gpt_transformer,
                            build_lr_performer,]
                            # build_lr_reformer,
                            # build_lr_xformer]
    for builder in transformer_builders:
        model = builder(**kwargs)
        start = time()
        args = torch.randint(0, 8192, (16,450))
        for k in tqdm(range(10)):
            model(args)
        stop = time()
        print(f"Model: {str(builder)}; Elapsed: {stop-start}")


if __name__ == '__main__':
    test_all_performance(layers=12, model_dim=512, heads=8, num_tokens=8192, max_seq_len=1000, checkpointing=False)