import torch
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
        return x


class ChainedEmbeddingGen(nn.Module):
    def __init__(self, depth=10):
        super(ChainedEmbeddingGen, self).__init__()
        self.initial_conv = ConvGnLelu(3, 64, kernel_size=7, bias=True, norm=False, activation=False)
        self.spine = SpineNet(arch='49', output_level=[3, 4], double_reduce_early=False)
        self.blocks = nn.ModuleList([BasicEmbeddingPyramid() for i in range(depth)])
        self.upsample = FinalUpsampleBlock2x(64)

    def forward(self, x):
        emb = checkpoint(self.spine, x)
        fea = self.initial_conv(x)
        for block in self.blocks:
            fea = fea + checkpoint(block, fea, *emb)
        return checkpoint(self.upsample, fea),


class ChainedEmbeddingGenWithStructure(nn.Module):
    def __init__(self, depth=10, recurrent=False):
        super(ChainedEmbeddingGenWithStructure, self).__init__()
        self.recurrent = recurrent
        self.initial_conv = ConvGnLelu(3, 64, kernel_size=7, bias=True, norm=False, activation=False)
        if recurrent:
            self.recurrent_process = ConvGnLelu(3, 64, kernel_size=3, stride=2, norm=False, bias=True, activation=False)
            self.recurrent_join = ReferenceJoinBlock(64, residual_weight_init_factor=.01, final_norm=False, kernel_size=1, depth=3, join=False)
        self.spine = SpineNet(arch='49', output_level=[3, 4], double_reduce_early=False)
        self.blocks = nn.ModuleList([BasicEmbeddingPyramid() for i in range(depth)])
        self.structure_joins = nn.ModuleList([ConjoinBlock(64) for i in range(3)])
        self.structure_blocks = nn.ModuleList([ConvGnLelu(64, 64, kernel_size=3, bias=False, norm=False, activation=False, weight_init_factor=.1) for i in range(3)])
        self.structure_upsample = FinalUpsampleBlock2x(64)
        self.grad_extract = ImageGradientNoPadding()
        self.upsample = FinalUpsampleBlock2x(64)

    def forward(self, x, recurrent=None):
        emb = checkpoint(self.spine, x)
        fea = self.initial_conv(x)
        if self.recurrent:
            rec = self.recurrent_process(recurrent)
            fea, _ = self.recurrent_join(fea, rec)
        grad = fea
        for i, block in enumerate(self.blocks):
            fea = fea + checkpoint(block, fea, *emb)
            if i < 3:
                structure_br = checkpoint(self.structure_joins[i], grad, fea)
                grad = grad + checkpoint(self.structure_blocks[i], structure_br)
        out = checkpoint(self.upsample, fea)
        return out, self.grad_extract(checkpoint(self.structure_upsample, grad)), self.grad_extract(out)
