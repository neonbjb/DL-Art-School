import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.nn import timestep_embedding
from models.lucidrains.vq import VectorQuantize
from models.lucidrains.x_transformers import FeedForward, Attention, Decoder, RMSScaleShiftNorm
from trainer.networks import register_model
from utils.util import checkpoint


class SelfClassifyingHead(nn.Module):
    def __init__(self, dim, classes, out_dim, head_depth, seq_len, dropout, init_temperature):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = classes
        self.temperature = init_temperature
        self.dec = Decoder(dim=dim, depth=head_depth, heads=4, ff_dropout=dropout, ff_mult=2, attn_dropout=dropout,
                                                use_rmsnorm=True, ff_glu=True, do_checkpointing=False)
        self.quantizer = VectorQuantize(out_dim, classes, use_cosine_sim=False, threshold_ema_dead_code=2,
                                        sample_codebook_temp=init_temperature)
        self.to_output = nn.Linear(dim, out_dim)
        self.to_decoder = nn.Linear(out_dim, dim)

    def do_ar_step(self, x, used_codes):
        h = self.dec(x)
        o = self.to_output(h[:, -1])
        q, c, _ = self.quantizer(o, used_codes)
        return q, c

    def forward(self, x, target):
        # manually perform ar regression over sequence_length=self.seq_len
        stack = [x]
        outputs = []
        results = []
        codes = []
        q_reg = 0
        for i in range(self.seq_len):
            q, c = checkpoint(functools.partial(self.do_ar_step, used_codes=codes), torch.stack(stack, dim=1))
            q_reg = q_reg + (q ** 2).mean()
            s = torch.sigmoid(q)

            outputs.append(s)
            output = torch.stack(outputs, dim=1).sum(1)

            # If the addition would strictly make the result worse, set it to 0. Sometimes.
            if len(results) > 0:
                worsen = (F.mse_loss(outputs[-1], target, reduction='none').sum(-1) < F.mse_loss(output, target, reduction='none').sum(-1)).float()
                probabilistic_worsen = torch.rand_like(worsen) * worsen > .5
                output = output * probabilistic_worsen.unsqueeze(-1)  # This is non-differentiable, but still deterministic.
                c[probabilistic_worsen] = -1  # Code of -1 means the code was unused.
                s = s * probabilistic_worsen.unsqueeze(-1)
                outputs[-1] = s

            codes.append(c)
            stack.append(self.to_decoder(s))
            results.append(output)
        return results, torch.cat(codes, dim=0), q_reg / self.seq_len


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
        # Normalize x on [0,1]
        assert x.max() < 1.2 and x.min() > -1.2, f'{x.min()} {x.max()}'
        x = (x + 1) / 2

        b, c, s = x.shape
        px = x.permute(0,2,1)  # B,S,C shape
        f = px.reshape(-1, self.op_dim)
        h = self.proj(f)
        for lyr in self.encoder:
            h = lyr(h)

        reconstructions, codes, q_reg = self.heads(h, f)
        reconstruction_losses = torch.stack([F.mse_loss(r.reshape(b, s, c), px) for r in reconstructions])
        r_follow = torch.arange(1, reconstruction_losses.shape[0]+1, device=x.device)
        reconstruction_losses = (reconstruction_losses * r_follow / r_follow.shape[0])
        self.log_codes(codes)

        return reconstruction_losses, q_reg

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
                    'temperature': self.heads.quantizer._codebook.sample_codebook_temp}
        else:
            return {}

    def update_for_step(self, step, *args):
        self.internal_step = step
        self.heads.quantizer._codebook.sample_codebook_temp = max(
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
    inp = torch.randn((4,256,200)).clamp(-1,1)
    model = InstrumentQuantizer(256, 512, 4096, 8, 3)
    model(inp)
