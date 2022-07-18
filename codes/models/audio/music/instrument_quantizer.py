import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.nn import timestep_embedding
from models.lucidrains.x_transformers import FeedForward, Attention, Decoder, RMSScaleShiftNorm
from trainer.networks import register_model
from utils.util import checkpoint


class SelfClassifyingHead(nn.Module):
    def __init__(self, dim, classes, out_dim, head_depth, seq_len, dropout, init_temperature):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = classes
        self.temperature = init_temperature
        self.dec = Decoder(dim=dim, depth=head_depth, heads=2, ff_dropout=dropout, ff_mult=2, attn_dropout=dropout,
                                                use_rmsnorm=True, ff_glu=True, rotary_pos_emb=True)
        self.to_classes = nn.Linear(dim, classes)
        self.feedback_codebooks = nn.Linear(classes, dim, bias=False)
        self.codebooks = nn.Linear(classes, out_dim, bias=False)

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

    def do_ar_step(self, x, used_codes):
        h = self.dec(x)
        h = self.to_classes(h[:,-1])
        for uc in used_codes:
            mask = torch.arange(0, self.num_classes, device=x.device).unsqueeze(0).repeat(x.shape[0],1) == uc.unsqueeze(1)
            h[mask] = -torch.inf
        c = F.gumbel_softmax(h, tau=self.temperature, hard=self.temperature==1)\

        soft_c = torch.softmax(h, dim=-1)
        perplexity = self._compute_perplexity(soft_c)
        return c, perplexity

    def forward(self, x):
        with torch.no_grad():
            # Force one of the codebook weights to zero, allowing the model to "skip" any classes it chooses.
            self.codebooks.weight.data[:, 0] = 0

        # manually perform ar regression over sequence_length=self.seq_len
        stack = [x]
        outputs = []
        results = []
        codes = []
        total_perplexity = 0
        for i in range(self.seq_len):
            nc, perp = checkpoint(functools.partial(self.do_ar_step, used_codes=codes), torch.stack(stack, dim=1))
            codes.append(nc.argmax(-1))
            stack.append(self.feedback_codebooks(nc))
            outputs.append(self.codebooks(nc))
            results.append(torch.stack(outputs, dim=1).sum(1))
            total_perplexity = total_perplexity + perp
        return results, total_perplexity / self.seq_len, torch.cat(codes, dim=-1)


class VectorResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.ff = FeedForward(dim, mult=2, glu=True, dropout=dropout, zero_init_output=True)

    def forward(self, x):
        h = self.norm(x.unsqueeze(-1)).squeeze(-1)
        h = self.ff(h)
        return h + x


class InstrumentQuantizer(nn.Module):
    def __init__(self, op_dim, dim, num_classes, enc_depth, head_depth, class_seq_len=5, dropout=.1,
                 min_temp=1, max_temp=10, temp_decay=.999):
        """

        Args:
            op_dim:
            dim:
            num_classes:
            enc_depth:
            head_depth:
            class_seq_len:
            dropout:
            min_temp:
            max_temp:
            temp_decay: Temperature decay. Default value of .999 decays ~50% in 1000 steps.
        """
        super().__init__()
        self.op_dim = op_dim
        self.proj = nn.Linear(op_dim, dim)
        self.encoder = nn.ModuleList([VectorResBlock(dim, dropout) for _ in range(enc_depth)])
        self.heads = SelfClassifyingHead(dim, num_classes, op_dim, head_depth, class_seq_len, dropout, max_temp)
        self.min_gumbel_temperature = min_temp
        self.max_gumbel_temperature = max_temp
        self.gumbel_temperature_decay = temp_decay

        self.codes = torch.zeros((3000000,), dtype=torch.long)
        self.internal_step = 0
        self.code_ind = 0
        self.total_codes = 0

    def forward(self, x):
        b, c, s = x.shape
        px = x.permute(0,2,1)  # B,S,C shape
        f = px.reshape(-1, self.op_dim)
        h = self.proj(f)
        for lyr in self.encoder:
            h = lyr(h)

        reconstructions, perplexity, codes = self.heads(h)
        reconstruction_losses = torch.stack([F.mse_loss(r.reshape(b, s, c), px) for r in reconstructions])
        r_follow = torch.arange(1, reconstruction_losses.shape[0]+1, device=x.device)
        reconstruction_losses = (reconstruction_losses * r_follow / r_follow.shape[0])
        self.log_codes(codes)

        return reconstruction_losses, perplexity

    def log_codes(self, codes):
        if self.internal_step % 5 == 0:
            l = codes.shape[0]
            i = self.code_ind if (self.codes.shape[0] - self.code_ind) > l else self.codes.shape[0] - l
            self.codes[i:i+l] = codes.cpu()
            self.code_ind = self.code_ind + l
            if self.code_ind >= self.codes.shape[0]:
                self.code_ind = 0
            self.total_codes += 1

    def get_debug_values(self, step, __):
        if self.total_codes > 0:
            return {'histogram_codes': self.codes[:self.total_codes],
                    'temperature': self.heads.temperature}
        else:
            return {}

    def update_for_step(self, step, *args):
        self.internal_step = step
        self.heads.temperature = max(
                    self.max_gumbel_temperature * self.gumbel_temperature_decay**step,
                    self.min_gumbel_temperature,
                )

    def get_grad_norm_parameter_groups(self):
        groups = {
            'encoder': list(self.encoder.parameters()),
            'heads': list(self.heads.parameters()),
            'proj': list(self.proj.parameters()),
        }
        return groups


@register_model
def register_instrument_quantizer(opt_net, opt):
    return InstrumentQuantizer(**opt_net['kwargs'])


if __name__ == '__main__':
    inp = torch.randn((4,256,200))
    model = InstrumentQuantizer(256, 512, 4096, 8, 3)
    model(inp)
