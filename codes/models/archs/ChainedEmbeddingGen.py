import os

import torch
import torchvision
from torch import nn

from models.archs.SPSR_arch import ImageGradientNoPadding
from models.archs.arch_util import ConvGnLelu, ExpansionBlock2, ConvGnSilu, ConjoinBlock, MultiConvBlock, \
    FinalUpsampleBlock2x, ReferenceJoinBlock
from models.archs.spinenet_arch import SpineNet
from utils.util import checkpoint


class BasicEmbeddingPyramid(nn.Module):
    def __init__(self, use_norms=True):
        super(BasicEmbeddingPyramid, self).__init__()
        self.initial_process = ConvGnLelu(64, 64, kernel_size=1, bias=True, activation=True, norm=False)
        self.reducers = nn.ModuleList([ConvGnLelu(64, 128, stride=2, kernel_size=1, bias=False, activation=True, norm=False),
                                  ConvGnLelu(128, 128, kernel_size=3, bias=False, activation=True, norm=use_norms),
                                  ConvGnLelu(128, 256, stride=2, kernel_size=1, bias=False, activation=True, norm=False),
                                  ConvGnLelu(256, 256, kernel_size=3, bias=False, activation=True, norm=use_norms)])
        self.expanders = nn.ModuleList([ExpansionBlock2(256, 128, block=ConvGnLelu),
                                   ExpansionBlock2(128, 64, block=ConvGnLelu)])
        self.embedding_processor1 = ConvGnSilu(256, 128, kernel_size=1, bias=True, activation=True, norm=False)
        self.embedding_joiner1 = ConjoinBlock(128, block=ConvGnLelu, norm=use_norms)
        self.embedding_processor2 = ConvGnSilu(256, 256, kernel_size=1, bias=True, activation=True, norm=False)
        self.embedding_joiner2 = ConjoinBlock(256, block=ConvGnLelu, norm=use_norms)

        self.final_process = nn.Sequential(ConvGnLelu(128, 96, kernel_size=1, bias=False, activation=False, norm=False,
                                                      weight_init_factor=.1),
                                           ConvGnLelu(96, 64, kernel_size=1, bias=False, activation=False, norm=False,
                                                      weight_init_factor=.1),
                                           ConvGnLelu(64, 64, kernel_size=1, bias=False, activation=False, norm=False,
                                                      weight_init_factor=.1),
                                           ConvGnLelu(64, 64, kernel_size=1, bias=False, activation=False, norm=False,
                                                      weight_init_factor=.1))

    def forward(self, x, *embeddings):
        p = self.initial_process(x)
        identities = []
        for i in range(2):
            identities.append(p)
            p = self.reducers[i*2](p)
            p = self.reducers[i*2+1](p)
            if i == 0:
                p = self.embedding_joiner1(p, self.embedding_processor1(embeddings[0]))
            elif i == 1:
                p = self.embedding_joiner2(p, self.embedding_processor2(embeddings[1]))
        for i in range(2):
            p = self.expanders[i](p, identities[-(i+1)])
        x = self.final_process(torch.cat([x, p], dim=1))
        return x, p


class MultifacetedChainedEmbeddingGen(nn.Module):
    def __init__(self, depth=10, scale=2):
        super(MultifacetedChainedEmbeddingGen, self).__init__()
        assert scale == 2 or scale == 4

        self.initial_conv = ConvGnLelu(3, 64, kernel_size=7, bias=True, norm=False, activation=False)

        self.teco_recurrent_process = ConvGnLelu(3, 64, kernel_size=3, stride=2, norm=False, bias=True, activation=False)
        self.teco_recurrent_join = ReferenceJoinBlock(64, residual_weight_init_factor=.01, final_norm=False, kernel_size=1, depth=3, join=False)

        self.prog_recurrent_process = ConvGnLelu(64, 64, kernel_size=3, stride=1, norm=False, bias=True, activation=False)
        self.prog_recurrent_join = ReferenceJoinBlock(64, residual_weight_init_factor=.01, final_norm=False, kernel_size=1, depth=3, join=False)

        self.spine = SpineNet(arch='49', output_level=[3, 4], double_reduce_early=False)
        self.blocks = nn.ModuleList([BasicEmbeddingPyramid() for i in range(depth)])
        self.bypasses = nn.ModuleList([OptionalPassthroughBlock(64, initial_bias=0) for i in range(depth)])
        self.structure_joins = nn.ModuleList([ConjoinBlock(64) for i in range(3)])
        self.structure_blocks = nn.ModuleList([ConvGnLelu(64, 64, kernel_size=3, bias=False, norm=False, activation=False, weight_init_factor=.1) for i in range(3)])
        self.structure_upsample = FinalUpsampleBlock2x(64, scale=scale)
        self.grad_extract = ImageGradientNoPadding()
        self.upsample = FinalUpsampleBlock2x(64, scale=scale)

        self.teco_ref_std = 0
        self.prog_ref_std = 0
        self.block_residual_means = [0 for _ in range(depth)]
        self.block_residual_stds = [0 for _ in range(depth)]
        self.bypass_maps = []

    def forward(self, x, teco_recurrent=None, prog_recurrent=None):
        fea = self.initial_conv(x)

        # Integrate recurrence inputs.
        if teco_recurrent is not None:
            teco_rec = self.teco_recurrent_process(teco_recurrent)
            fea, std = self.teco_recurrent_join(fea, teco_rec)
            self.teco_ref_std = std.item()
        elif prog_recurrent is not None:
            prog_rec = self.prog_recurrent_process(prog_recurrent)
            prog_rec, std = self.prog_recurrent_join(fea, prog_rec)
            self.prog_ref_std = std.item()

        emb = checkpoint(self.spine, fea)
        grad = fea
        self.bypass_maps = []
        for i, block in enumerate(self.blocks):
            residual, context = checkpoint(block, fea, *emb)
            residual, bypass_map = checkpoint(self.bypasses[i], residual, context)
            fea = fea + residual
            self.bypass_maps.append(bypass_map.detach())
            self.block_residual_means[i] = residual.mean().item()
            self.block_residual_stds[i] = residual.std().item()
            if i < 3:
                structure_br = checkpoint(self.structure_joins[i], grad, fea)
                grad = grad + checkpoint(self.structure_blocks[i], structure_br)
        out = checkpoint(self.upsample, fea)
        return out, self.grad_extract(checkpoint(self.structure_upsample, grad)), self.grad_extract(out), fea

    def visual_dbg(self, step, path):
        for i, bm in enumerate(self.bypass_maps):
            torchvision.utils.save_image(bm.cpu().float(), os.path.join(path, "%i_bypass_%i.png" % (step, i+1)))

    def get_debug_values(self, step, net_name):
        biases = [b.bias.item() for b in self.bypasses]
        blk_stds, blk_means = {}, {}
        for i, (s, m) in enumerate(zip(self.block_residual_stds, self.block_residual_means)):
            blk_stds['block_%i' % (i+1,)] = s
            blk_means['block_%i' % (i+1,)] = m
        return {'teco_std': self.teco_ref_std,
                'prog_std': self.prog_ref_std,
                'bypass_biases': sum(biases) / len(biases),
                'blocks_std': blk_stds, 'blocks_mean': blk_means}
