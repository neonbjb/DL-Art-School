import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2Model

from models.arch_util import AttentionBlock, ResBlock
from models.audio.tts.lucidrains_dvae import DiscreteVAE
from models.lucidrains.x_transformers import Encoder
from trainer.networks import register_model
from utils.util import opt_get, ceil_multiple, print_network


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 cond_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=8,
                 dropout=.1,
                 do_checkpointing=False):
        super().__init__()
        self.init = nn.Conv1d(cond_dim, embedding_dim, kernel_size=1)
        self.attn = Encoder(
                dim=embedding_dim,
                depth=attn_blocks,
                heads=num_attn_heads,
                ff_dropout=dropout,
                attn_dropout=dropout,
                use_rmsnorm=True,
                ff_glu=True,
                rotary_pos_emb=True,
                zero_init_branch_output=True,
                ff_mult=2,
                do_checkpointing=do_checkpointing
            )
        self.dim = embedding_dim

    def forward(self, x):
        h = self.init(x).permute(0,2,1)
        h = self.attn(h).permute(0,2,1)
        return h.mean(-1)


class ConditioningAR(nn.Module):
    def __init__(self, dim, layers, dropout=0, num_vectors=8192, cond_free_percent=.15, fp16=False):
        super().__init__()
        self.cond_encoder = ConditioningEncoder(256, dim)
        self.cond_free_emb = nn.Parameter(torch.randn(1,dim))
        self.unconditioned_percentage = cond_free_percent
        self.fp16 = fp16

        self.config = GPT2Config(vocab_size=1, n_positions=8192, n_embd=dim, n_layer=layers, n_head=dim//64,
                                 n_inner=dim*2, attn_pdrop=dropout, resid_pdrop=dropout, gradient_checkpointing=True,
                                 use_cache=False)
        self.gpt = GPT2Model(self.config)
        del self.gpt.wte  # Unused, we'll do our own embeddings.

        self.embeddings = nn.Embedding(num_vectors, dim)
        self.head = nn.Linear(dim, num_vectors)

    def forward(self, cheater_codes, conditioning, code_lengths=None, return_latent=False):
        unused_params = []

        cond = self.cond_encoder(conditioning)
        if self.training and self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand((cond.shape[0],1), device=cond.device) < self.unconditioned_percentage
            cond = torch.where(unconditioned_batches, self.cond_free_emb.repeat(cond.shape[0],1), cond)
            unused_params.append(self.cond_free_emb)

        h = self.embeddings(cheater_codes)
        h = torch.cat([cond.unsqueeze(1), h], dim=1)
        targets = cheater_codes  # Since we padded above by 1, the input alignment works.

        with torch.autocast(cheater_codes.device.type, enabled=self.fp16):
            h = self.gpt(inputs_embeds=h, return_dict=True).last_hidden_state

            if return_latent:
                return h.float()

            logits = self.head(h[:,:-1]).permute(0,2,1)
            loss = F.cross_entropy(logits, targets, reduction="none")

        # Perform masking
        if code_lengths is not None:
            mask = torch.arange(0, loss.shape[1], device=h.device).unsqueeze(0).repeat(loss.shape[0], 1) < code_lengths.unsqueeze(1)
            loss = loss * mask
        loss = loss.mean()

        unused_adder = 0
        for p in unused_params:
            unused_adder = unused_adder + p.mean() * 0
        loss = loss + unused_adder

        return loss

    def get_grad_norm_parameter_groups(self):
        groups = {
            'gpt': list(self.gpt.parameters()),
            'head': list(self.head.parameters()),
            'embeddings': list(self.embeddings.parameters()),
            'conditioning_encoder': list(self.cond_encoder.parameters()),
        }
        return groups


@register_model
def register_cheater_gen_ar(opt_net, opt):
    return ConditioningAR(**opt_get(opt_net, ['kwargs'], {}))


def test_ar():
    model = ConditioningAR(512, 8, cond_free_percent=.5)
    print_network(model)

    codes = torch.randint(0,8192, (2,400))
    cond = torch.randn(2,256,400)
    cl = torch.tensor([200,10])
    codes[1,10:] = 2
    model(codes, cond, cl)
    pg = model.get_grad_norm_parameter_groups()



if __name__ == '__main__':
    test_ar()
