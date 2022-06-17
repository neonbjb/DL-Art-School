import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2Model

from models.arch_util import AttentionBlock, ResBlock
from models.audio.tts.lucidrains_dvae import DiscreteVAE
from trainer.networks import register_model
from utils.util import opt_get, ceil_multiple, print_network


class UpperEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 hidden_dim,
                 embedding_dim,
                 ):
        super().__init__()
        attn = []
        def edim(m):
            dd = min(spec_dim + m * 128, hidden_dim)
            return ceil_multiple(dd, 8)
        self.downsampler = nn.Sequential(
            ResBlock(spec_dim, out_channels=edim(1), use_conv=True, dims=1, down=True),
            ResBlock(edim(1), out_channels=edim(2), use_conv=True, dims=1, down=True),
            ResBlock(edim(2), out_channels=edim(3), use_conv=True, dims=1, down=True),
            ResBlock(edim(3), out_channels=edim(4), use_conv=True, dims=1),
            ResBlock(edim(4), out_channels=hidden_dim, use_conv=True, dims=1, down=True))
        self.encoder = nn.Sequential(
            AttentionBlock(hidden_dim, 4, do_activation=True),
            ResBlock(hidden_dim, out_channels=hidden_dim, use_conv=True, dims=1),
            AttentionBlock(hidden_dim, 4, do_activation=True),
            ResBlock(hidden_dim, out_channels=hidden_dim, use_conv=True, dims=1),
            AttentionBlock(hidden_dim, 4, do_activation=True),
            ResBlock(hidden_dim, out_channels=hidden_dim, use_conv=True, dims=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, embedding_dim, 1)
        )

    def forward(self, x):
        h = self.downsampler(x)
        h = self.encoder(h)
        return h


class GptMusicLower(nn.Module):
    def __init__(self, dim, layers, encoder_out_dim, dropout=0, num_target_vectors=8192, fp16=True, num_vaes=4, vqargs={}):
        super().__init__()
        self.num_vaes = num_vaes
        self.start_token = nn.Parameter(torch.randn(1, 1, dim))
        self.config = GPT2Config(vocab_size=1, n_positions=8192, n_embd=dim, n_layer=layers, n_head=dim//64,
                                 n_inner=dim*2, attn_pdrop=dropout, resid_pdrop=dropout, gradient_checkpointing=True,
                                 use_cache=False)

        self.target_quantizers = nn.ModuleList([DiscreteVAE(**vqargs).eval() for _ in range(num_vaes)])
        self.upper_encoder = UpperEncoder(256, dim, encoder_out_dim)
        self.encoder_projector = nn.Conv1d(encoder_out_dim, dim, 1)
        self.fp16 = fp16

        # Freeze the target quantizer.
        for p in self.target_quantizers.parameters():
            p.DO_NOT_TRAIN = True
            p.requires_grad = False
        # And delete the decoder, which is unused.
        for tq in self.target_quantizers:
            del tq.decoder

        self.gpt = GPT2Model(self.config)
        del self.gpt.wte  # Unused, we'll do our own embeddings.

        self.embeddings = nn.ModuleList([nn.Embedding(num_target_vectors, dim // num_vaes) for _ in range(num_vaes)])
        self.heads = nn.ModuleList([nn.Linear(dim, num_target_vectors) for _ in range(num_vaes)])

    def forward(self, mel, return_latent=False):
        unused_params = []

        with torch.no_grad():
            codes = []
            partition_size = mel.shape[1] // len(self.target_quantizers)
            for i, q in enumerate(self.target_quantizers):
                mel_partition = mel[:, i*partition_size:(i+1)*partition_size]
                codes.append(q.get_codebook_indices(mel_partition))
            codes = torch.stack(codes, dim=-1)

        upper_vector = self.upper_encoder(mel)
        upper_vector = self.encoder_projector(upper_vector)
        # WTB slerp
        upper_vector = F.interpolate(upper_vector, size=codes.shape[1], mode='linear')
        upper_vector = upper_vector.permute(0,2,1)

        inputs = codes[:, :-1]
        targets = codes
        upper_vector = upper_vector[:, :-1]
        h = [embedding(inputs[:, :, i]) for i, embedding in enumerate(self.embeddings)]
        h = torch.cat(h, dim=-1) + upper_vector

        with torch.autocast(mel.device.type, enabled=self.fp16):
            # Stick the conditioning embedding on the front of the input sequence.
            # The transformer will learn how to integrate it.
            # This statement also serves to pre-pad the inputs by one token, which is the basis of the next-token-prediction task. IOW: this is the "START" token.
            h = torch.cat([self.start_token.repeat(h.shape[0], 1, 1), h], dim=1)

            h = self.gpt(inputs_embeds=h, return_dict=True).last_hidden_state

            if return_latent:
                return h.float()

            losses = 0
            for i, head in enumerate(self.heads):
                logits = head(h).permute(0,2,1)
                loss = F.cross_entropy(logits, targets[:,:,i])
                losses = losses + loss

        unused_adder = 0
        for p in unused_params:
            unused_adder = unused_adder + p.mean() * 0
        losses = losses + unused_adder

        return losses / self.num_vaes

    def get_grad_norm_parameter_groups(self):
        groups = {
            'gpt': list(self.gpt.parameters()),
            'heads': list(self.heads.parameters()),
            'embeddings': list(self.embeddings.parameters()),
            'upper_latent_encoder': list(self.upper_encoder.encoder.parameters()),
            'upper_latent_downsampler': list(self.upper_encoder.downsampler.parameters()),
        }
        return groups



@register_model
def register_music_gpt_lower2(opt_net, opt):
    return GptMusicLower(**opt_get(opt_net, ['kwargs'], {}))


def test_lower():
    model = GptMusicLower(dim=1024, encoder_out_dim=256, layers=16, fp16=False, num_target_vectors=8192, num_vaes=4,
                          vqargs= {'positional_dims': 1, 'channels': 64,
            'hidden_dim': 512, 'num_resnet_blocks': 3, 'codebook_dim': 512, 'num_tokens': 8192,
            'num_layers': 0, 'record_codes': True, 'kernel_size': 3, 'use_transposed_convs': False,
                                                })
    quants = ['X:\\dlas\\experiments\\music_vqvaes\\train_lrdvae_music_low\\models\\7500_generator.pth',
              'X:\\dlas\\experiments\\music_vqvaes\\train_lrdvae_music_mid_low\\models\\11000_generator.pth',
              'X:\\dlas\\experiments\\music_vqvaes\\train_lrdvae_music_mid_high\\models\\11500_generator.pth',
              'X:\\dlas\\experiments\\music_vqvaes\\train_lrdvae_music_high\\models\\11500_generator.pth']
    for i, qfile in enumerate(quants):
        quant_weights = torch.load(qfile)
        model.target_quantizers[i].load_state_dict(quant_weights, strict=False)
    torch.save(model.state_dict(), 'sample.pth')
    print_network(model)

    mel = torch.randn(2,256,400)
    model(mel)
    pg = model.get_grad_norm_parameter_groups()

    t = 0
    for k, vs in pg.items():
        s = 0
        for v in vs:
            m = 1
            for d in v.shape:
                m *= d
            s += m
        t += s
        print(k, s/1000000)
    print(t/1000000)


if __name__ == '__main__':
    test_lower()