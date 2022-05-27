import torch
import torch.nn as nn
import torch.nn.functional as F

from models.arch_util import ResBlock, AttentionBlock
from models.audio.music.flat_diffusion import MultiGroupEmbedding
from trainer.networks import register_model
from utils.util import checkpoint


class Code2Mel(nn.Module):
    def __init__(self, out_dim=256, base_dim=1024, num_tokens=16, num_groups=4, dropout=.1):
        super().__init__()
        self.emb = MultiGroupEmbedding(num_tokens, num_groups, base_dim)
        self.base_blocks = nn.Sequential(ResBlock(base_dim, dropout, dims=1),
                                         AttentionBlock(base_dim, num_heads=base_dim//64),
                                         ResBlock(base_dim, dropout, dims=1))
        l2dim = base_dim-256
        self.l2_up_block = nn.Conv1d(base_dim, l2dim, kernel_size=5, padding=2)
        self.l2_blocks = nn.Sequential(ResBlock(l2dim, dropout, kernel_size=5, dims=1),
                                         AttentionBlock(l2dim, num_heads=base_dim//64),
                                         ResBlock(l2dim, dropout, kernel_size=5, dims=1),
                                         AttentionBlock(l2dim, num_heads=base_dim//64),
                                         ResBlock(l2dim, dropout, dims=1),
                                         ResBlock(l2dim, dropout, dims=1))
        l3dim = l2dim-256
        self.l3_up_block = nn.Conv1d(l2dim, l3dim, kernel_size=5, padding=2)
        self.l3_blocks = nn.Sequential(ResBlock(l3dim, dropout, kernel_size=5, dims=1),
                                       AttentionBlock(l3dim, num_heads=base_dim//64),
                                       ResBlock(l3dim, dropout, kernel_size=5, dims=1),
                                       ResBlock(l3dim, dropout, dims=1))
        self.final_block = nn.Conv1d(l3dim, out_dim, kernel_size=3, padding=1)

    def forward(self, codes, target):
        with torch.autocast(codes.device.type):
            h = self.emb(codes).permute(0,2,1)
            h = checkpoint(self.base_blocks, h)
            h = F.interpolate(h, scale_factor=2, mode='linear')
            h = self.l2_up_block(h)
            h = checkpoint(self.l2_blocks, h)
            h = F.interpolate(h, size=target.shape[-1], mode='linear')
            h = self.l3_up_block(h)
        h = checkpoint(self.l3_blocks, h.float())
        pred = self.final_block(h)
        return F.mse_loss(pred, target), pred


@register_model
def register_code2mel(opt_net, opt):
    return Code2Mel(**opt_net['kwargs'])


if __name__ == '__main__':
    model = Code2Mel()
    codes = torch.randint(0,16, (2,200,4))
    target = torch.randn(2,256,804)
    model(codes, target)