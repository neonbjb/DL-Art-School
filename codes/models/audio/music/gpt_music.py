import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from models.arch_util import AttentionBlock, ResBlock
from models.audio.music.music_quantizer import MusicQuantizer
from models.audio.music.music_quantizer2 import MusicQuantizer2
from models.audio.tts.lucidrains_dvae import DiscreteVAE
from models.lucidrains.x_transformers import Encoder
from models.vqvae.vqvae import Quantize
from trainer.networks import register_model
from utils.util import opt_get, checkpoint, ceil_multiple, print_network


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=3, stride=2, padding=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads, do_activation=True))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        return h.mean(dim=2)


class UpperConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4):
        super().__init__()
        attn = []
        self.init = nn.Sequential(nn.Conv1d(spec_dim, min(spec_dim+128, embedding_dim), kernel_size=3, stride=2, padding=1),
                                  nn.Conv1d(min(spec_dim+128, embedding_dim), min(spec_dim+256, embedding_dim), kernel_size=3, stride=2, padding=1),
                                  nn.Conv1d(min(spec_dim+256, embedding_dim), min(spec_dim+384, embedding_dim), kernel_size=3, stride=2, padding=1),
                                  nn.Conv1d(min(spec_dim+384, embedding_dim), min(spec_dim+512, embedding_dim), kernel_size=3, stride=2, padding=1),
                                  ResBlock(min(spec_dim+512, embedding_dim), dims=1),
                                  nn.Conv1d(min(spec_dim+512, embedding_dim), min(spec_dim+512, embedding_dim), kernel_size=3, stride=2, padding=1),
                                  ResBlock(min(spec_dim+512, embedding_dim), dims=1))
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads, do_activation=True))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        return h.mean(dim=2)


class UpperQuantizer(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 num_tokens):
        super().__init__()
        attn = []
        def edim(m):
            dd = max(embedding_dim//m, 128, spec_dim)
            return ceil_multiple(dd, 8)
        self.encoder = nn.Sequential(
            ResBlock(spec_dim, out_channels=edim(6), use_conv=True, dims=1, down=True),
            ResBlock(edim(6), out_channels=edim(5), use_conv=True, dims=1, down=True),
            ResBlock(edim(5), out_channels=edim(4), use_conv=True, dims=1, down=True),
            ResBlock(edim(4), out_channels=edim(3), use_conv=True, dims=1, down=True),
            ResBlock(edim(3), out_channels=edim(3), use_conv=True, dims=1),
            ResBlock(edim(3), out_channels=edim(2), use_conv=True, dims=1, down=True),
            ResBlock(edim(2), out_channels=edim(2), use_conv=True, dims=1),
            ResBlock(edim(2), out_channels=embedding_dim, use_conv=True, dims=1, down=True),
            ResBlock(embedding_dim, out_channels=embedding_dim, use_conv=True, dims=1),
            ResBlock(embedding_dim, out_channels=embedding_dim, use_conv=True, dims=1),
            ResBlock(embedding_dim, out_channels=embedding_dim, use_conv=True, dims=1),
            nn.GroupNorm(8, embedding_dim)
        )
        self.quantizer = Quantize(embedding_dim, num_tokens)

        self.codes = torch.zeros((num_tokens*100,), dtype=torch.long)
        self.code_ind = 0
        self.total_codes = 0
        self.internal_step = 0

    def forward(self, x):
        h = x
        for lyr in self.encoder:
            h = lyr(h)
        h = h.permute(0,2,1)
        h_quant, commitment_loss, codes = self.quantizer(h)
        self.log_codes(codes)
        return h_quant, commitment_loss

    def log_codes(self, codes):
        # This is so we can debug the distribution of codes being learned.
        if self.internal_step % 10 == 0:
            codes = codes.flatten()
            l = codes.shape[0]
            i = self.code_ind if (self.codes.shape[0] - self.code_ind) > l else self.codes.shape[0] - l
            self.codes[i:i+l] = codes.cpu()
            self.code_ind = self.code_ind + l
            if self.code_ind >= self.codes.shape[0]:
                self.code_ind = 0
            self.total_codes += 1
        self.internal_step += 1


class GptMusicLower(nn.Module):
    def __init__(self, dim, layers, dropout=0, num_target_vectors=8192, num_upper_vectors=32768,
                 fp16=True, freeze_upper_until=0, num_vaes=4, vqargs={}):
        super().__init__()
        self.num_vaes = num_vaes
        self.freeze_upper_until = freeze_upper_until
        self.config = GPT2Config(vocab_size=1, n_positions=8192, n_embd=dim, n_layer=layers, n_head=dim//64,
                                 n_inner=dim*2, attn_pdrop=dropout, resid_pdrop=dropout, gradient_checkpointing=True, use_cache=False)
        self.target_quantizers = nn.ModuleList([DiscreteVAE(**vqargs).eval() for _ in range(num_vaes)])
        self.upper_quantizer = UpperQuantizer(256, dim, num_upper_vectors)
        self.fp16 = fp16
        self.internal_step = 0

        # Freeze the target quantizer.
        for p in self.target_quantizers.parameters():
            p.DO_NOT_TRAIN = True
            p.requires_grad = False

        self.conditioning_encoder = ConditioningEncoder(256, dim, attn_blocks=4, num_attn_heads=dim//64)

        self.gpt = GPT2Model(self.config)
        del self.gpt.wte  # Unused, we'll do our own embeddings.

        self.embeddings = nn.ModuleList([nn.Embedding(num_target_vectors, dim // num_vaes) for _ in range(num_vaes)])
        self.heads = nn.ModuleList([nn.Linear(dim, num_target_vectors) for _ in range(num_vaes)])

    def forward(self, mel, conditioning, return_latent=False):
        unused_params = []

        with torch.no_grad():
            codes = []
            partition_size = mel.shape[1] // len(self.target_quantizers)
            for i, q in enumerate(self.target_quantizers):
                mel_partition = mel[:, i*partition_size:(i+1)*partition_size]
                codes.append(q.get_codebook_indices(mel_partition))
            codes = torch.stack(codes, dim=-1)

        if self.freeze_upper_until > self.internal_step:
            with torch.no_grad():
                self.upper_quantizer = self.upper_quantizer.eval()
                upper_vector, upper_diversity = self.upper_quantizer(mel)
            unused_params.extend(list(self.upper_quantizer.parameters()))
        else:
            self.upper_quantizer = self.upper_quantizer.train()
            upper_vector, upper_diversity = self.upper_quantizer(mel, return_decoder_latent=True)
        upper_vector = F.interpolate(upper_vector.permute(0,2,1), size=codes.shape[1], mode='linear')
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
            cond_emb = self.conditioning_encoder(conditioning).unsqueeze(1)
            h = torch.cat([cond_emb, h], dim=1)

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

        return losses / self.num_vaes, upper_diversity

    def get_grad_norm_parameter_groups(self):
        groups = {
            'gpt': list(self.gpt.parameters()),
            'conditioning': list(self.conditioning_encoder.parameters()),
            'upper_quantizer': list(self.upper_quantizer.parameters()),
            'target_vqs': list(self.target_quantizers.parameters()),
        }
        return groups

    def get_debug_values(self, step, __):
        self.internal_step = 0
        if self.upper_quantizer.total_codes > 0:
            return {'histogram_upper_codes': self.upper_quantizer.codes[:self.upper_quantizer.total_codes]}
        else:
            return {}


class GptMusicUpper(nn.Module):
    def __init__(self, dim, layers, dropout=0, num_upper_vectors=64, num_upper_groups=4, fp16=True):
        super().__init__()
        self.internal_step = 0
        self.num_groups = num_upper_groups
        self.fp16 = fp16
        self.config = GPT2Config(vocab_size=1, n_positions=8192, n_embd=dim, n_layer=layers, n_head=dim//64,
                                 n_inner=dim*2, attn_pdrop=dropout, resid_pdrop=dropout, gradient_checkpointing=True,
                                 use_cache=False)
        self.upper_quantizer = MusicQuantizer2(inp_channels=256, inner_dim=[dim,
                                                                            max(512,dim-128),
                                                                            max(512,dim-256),
                                                                            max(512,dim-384),
                                                                            max(512,dim-512),
                                                                            max(512,dim-512)], codevector_dim=dim,
                                               codebook_size=num_upper_vectors, codebook_groups=num_upper_groups,
                                               expressive_downsamples=True)
        # Following are unused quantizer constructs we delete to avoid DDP errors (and to be efficient.. of course..)
        del self.upper_quantizer.up
        # Freeze the quantizer.
        for p in self.upper_quantizer.parameters():
            p.DO_NOT_TRAIN = True
            p.requires_grad = False

        self.conditioning_encoder = UpperConditioningEncoder(256, dim, attn_blocks=4, num_attn_heads=dim//64)

        self.gpt = GPT2Model(self.config)
        del self.gpt.wte  # Unused, we'll do our own embeddings.

        self.embeddings = nn.ModuleList([nn.Embedding(num_upper_vectors, dim // num_upper_groups) for _ in range(num_upper_groups)])
        self.heads = nn.ModuleList([nn.Linear(dim, num_upper_vectors) for _ in range(num_upper_groups)])


    def forward(self, mel, conditioning, return_latent=False):
        with torch.no_grad():
            self.upper_quantizer.eval()
            codes = self.upper_quantizer.get_codes(mel)

        inputs = codes[:, :-1]
        targets = codes
        h = [embedding(inputs[:, :, i]) for i, embedding in enumerate(self.embeddings)]
        h = torch.cat(h, dim=-1)

        with torch.autocast(mel.device.type, enabled=self.fp16):
            # Stick the conditioning embedding on the front of the input sequence.
            # The transformer will learn how to integrate it.
            # This statement also serves to pre-pad the inputs by one token, which is the basis of the next-token-prediction task. IOW: this is the "START" token.
            cond_emb = self.conditioning_encoder(conditioning).unsqueeze(1)
            h = torch.cat([cond_emb, h], dim=1)

            h = self.gpt(inputs_embeds=h, return_dict=True).last_hidden_state

            if return_latent:
                return h.float()

            losses = 0
            for i, head in enumerate(self.heads):
                logits = head(h).permute(0,2,1)
                loss = F.cross_entropy(logits, targets[:,:,i])
                losses = losses + loss

        return losses / self.num_groups

    def get_grad_norm_parameter_groups(self):
        groups = {
            'gpt': list(self.gpt.parameters()),
            'conditioning': list(self.conditioning_encoder.parameters()),
        }
        return groups

    def get_debug_values(self, step, __):
        if self.upper_quantizer.total_codes > 0:
            return {'histogram_upper_codes': self.upper_quantizer.codes[:self.upper_quantizer.total_codes]}
        else:
            return {}


@register_model
def register_music_gpt_lower(opt_net, opt):
    return GptMusicLower(**opt_get(opt_net, ['kwargs'], {}))

@register_model
def register_music_gpt_upper(opt_net, opt):
    return GptMusicUpper(**opt_get(opt_net, ['kwargs'], {}))


def test_lower():
    model = GptMusicLower(dim=512, layers=12, fp16=False, freeze_upper_until=1000,
                          num_target_vectors=8192, num_upper_vectors=8192, num_vaes=4,
                          vqargs= {
                                                     'positional_dims': 1, 'channels': 64,
            'hidden_dim': 512, 'num_resnet_blocks': 3, 'codebook_dim': 512, 'num_tokens': 8192,
            'num_layers': 0, 'record_codes': True, 'kernel_size': 3, 'use_transposed_convs': False,
                                                })
    quants = ['X:\\dlas\\experiments\\music_vqvaes\\train_lrdvae_music_low\\models\\7500_generator.pth',
              'X:\\dlas\\experiments\\music_vqvaes\\train_lrdvae_music_mid_low\\models\\11000_generator.pth',
              'X:\\dlas\\experiments\\music_vqvaes\\train_lrdvae_music_mid_high\\models\\11500_generator.pth',
              'X:\\dlas\\experiments\\music_vqvaes\\train_lrdvae_music_high\\models\\11500_generator.pth']
    for i, qfile in enumerate(quants):
        quant_weights = torch.load(qfile)
        model.target_quantizers[i].load_state_dict(quant_weights, strict=True)
    torch.save(model.state_dict(), 'sample.pth')
    print_network(model)

    mel = torch.randn(2,256,400)
    model(mel, mel)
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


def test_upper():
    lower = GptMusicLower(512, 12)
    lower.load_state_dict(torch.load('D:\\dlas\\experiments\\train_music_gpt\\models\\44500_generator_ema.pth'))
    model = GptMusicUpper(512, 12)
    model.upper_quantizer.load_state_dict(lower.upper_quantizer.state_dict())
    torch.save(model.state_dict(), 'sample.pth')
    mel = torch.randn(2,256,2500)
    model(mel, mel)
    model.get_grad_norm_parameter_groups()


if __name__ == '__main__':
    test_lower()