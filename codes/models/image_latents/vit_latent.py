import torch
import torch.nn as nn
import torch.nn.functional as F

from models.arch_util import ResBlock
from models.lucidrains.x_transformers import Encoder
from trainer.networks import register_model


class VitLatent(nn.Module):
    def __init__(self, top_dim, hidden_dim, depth, dropout=.1):
        super().__init__()
        self.upper = nn.Sequential(nn.Conv2d(3, top_dim, kernel_size=7, padding=3, stride=2),
                                   ResBlock(top_dim, use_conv=True, dropout=dropout),
                                   ResBlock(top_dim, out_channels=top_dim*2, down=True, use_conv=True, dropout=dropout),
                                   ResBlock(top_dim*2, use_conv=True, dropout=dropout),
                                   ResBlock(top_dim*2, out_channels=top_dim*4, down=True, use_conv=True, dropout=dropout),
                                   ResBlock(top_dim*4, use_conv=True, dropout=dropout),
                                   ResBlock(top_dim*4, out_channels=hidden_dim, down=True, use_conv=True, dropout=dropout),
                                   nn.GroupNorm(8, hidden_dim))
        self.encoder = Encoder(
                dim=hidden_dim,
                depth=depth,
                heads=hidden_dim//64,
                ff_dropout=dropout,
                attn_dropout=dropout,
                use_rmsnorm=True,
                ff_glu=True,
                rotary_pos_emb=True,
                ff_mult=2,
                do_checkpointing=True
            )

        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                 nn.BatchNorm1d(hidden_dim*2),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim*2, hidden_dim))

    def provide_ema(self, ema):
        self.ema = ema

    def project(self, x):
        h = self.upper(x)
        h = torch.flatten(h, 2).permute(0,2,1)
        h = self.encoder(h)[:,0]
        h_norm = F.normalize(h)
        return h_norm

    def forward(self, x1, x2):
        h1 = self.project(x1)
        #p1 = self.mlp(h1)
        h2 = self.project(x2)
        #p2 = self.mlp(h2)
        with torch.no_grad():
            he1 = self.ema.project(x1)
            he2 = self.ema.project(x2)

        def csim(h1, h2):
            b = x1.shape[0]
            sim = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(1).detach(), 2)
            eye = torch.eye(b, device=x1.device)
            neye = eye != 1
            return -(sim*eye).sum()/b, (sim*neye).sum()/(b**2-b)

        pos, neg = csim(h1, he2)
        pos2, neg2 = csim(h2, he1)
        return (pos+pos2)/2, (neg+neg2)/2

    def get_grad_norm_parameter_groups(self):
        return {
            'upper': list(self.upper.parameters()),
            'encoder': list(self.encoder.parameters()),
            'mlp': list(self.mlp.parameters()),
        }


@register_model
def register_vit_latent(opt_net, opt):
    return VitLatent(**opt_net['kwargs'])


if __name__ == '__main__':
    net = VitLatent(128, 1024, 8)
    net.provide_ema(net)
    x1 = torch.randn(2,3,244,244)
    x2 = torch.randn(2,3,244,244)
    net(x1,x2)
