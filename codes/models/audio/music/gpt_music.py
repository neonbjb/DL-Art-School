import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from models.arch_util import AttentionBlock, ResBlock
from models.audio.music.music_quantizer import MusicQuantizer
from models.audio.music.music_quantizer2 import MusicQuantizer2
from models.lucidrains.x_transformers import Encoder
from trainer.networks import register_model
from utils.util import opt_get, checkpoint


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
        h = checkpoint(self.init, x)
        h = self.attn(h
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


class GptMusicLower(nn.Module):
    def __init__(self, dim, layers, dropout=0, num_target_vectors=512, num_target_groups=2, num_upper_vectors=64, num_upper_groups=4, fp16=True):
        super().__init__()
        self.internal_step = 0
        self.num_groups = num_target_groups
        self.config = GPT2Config(vocab_size=1, n_positions=8192, n_embd=dim, n_layer=layers, n_head=dim//64,
                                 n_inner=dim*2, attn_pdrop=dropout, resid_pdrop=dropout, gradient_checkpointing=True, use_cache=False)
        self.target_quantizer = MusicQuantizer2(inp_channels=256, inner_dim=[1024], codevector_dim=1024, codebook_size=256,
                                                codebook_groups=2, max_gumbel_temperature=4, min_gumbel_temperature=.5)
        self.upper_quantizer = MusicQuantizer2(inp_channels=256, inner_dim=[dim,
                                                                            max(512,dim-128),
                                                                            max(512,dim-256),
                                                                            max(512,dim-384),
                                                                            max(512,dim-512),
                                                                            max(512,dim-512)], codevector_dim=dim,
                                               codebook_size=num_upper_vectors, codebook_groups=num_upper_groups, expressive_downsamples=True)
        self.fp16 = fp16
        # Following are unused quantizer constructs we delete to avoid DDP errors (and to be efficient.. of course..)
        del self.target_quantizer.decoder
        del self.target_quantizer.up
        del self.upper_quantizer.up
        # Freeze the target quantizer.
        for p in self.target_quantizer.parameters():
            p.DO_NOT_TRAIN = True
            p.requires_grad = False

        self.upper_mixer = Encoder(
                    dim=dim,
                    depth=4,
                    heads=dim//64,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_emb_dim=True,
                )
        self.conditioning_encoder = ConditioningEncoder(256, dim, attn_blocks=4, num_attn_heads=dim//64)

        self.gpt = GPT2Model(self.config)
        del self.gpt.wte  # Unused, we'll do our own embeddings.

        self.embeddings = nn.ModuleList([nn.Embedding(num_target_vectors, dim // num_target_groups) for _ in range(num_target_groups)])
        self.heads = nn.ModuleList([nn.Linear(dim, num_target_vectors) for _ in range(num_target_groups)])


    def forward(self, mel, conditioning, return_latent=False):
        with torch.no_grad():
            self.target_quantizer.eval()
            codes = self.target_quantizer.get_codes(mel)
        upper_vector, upper_diversity = self.upper_quantizer(mel, return_decoder_latent=True)
        upper_vector = self.upper_mixer(upper_vector.permute(0,2,1)).permute(0,2,1)  # Allow the upper vector to fully attend to itself (the whole thing is a prior.)
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

        return losses / self.num_groups, upper_diversity

    def get_grad_norm_parameter_groups(self):
        groups = {
            'gpt': list(self.gpt.parameters()),
            'conditioning': list(self.conditioning_encoder.parameters()),
            'upper_mixer': list(self.upper_mixer.parameters()),
            'upper_quant_down': list(self.upper_quantizer.down.parameters()),
            'upper_quant_encoder': list(self.upper_quantizer.encoder.parameters()),
            'upper_quant_codebook': [self.upper_quantizer.quantizer.codevectors],
        }
        return groups

    def get_debug_values(self, step, __):
        if self.upper_quantizer.total_codes > 0:
            return {'histogram_upper_codes': self.upper_quantizer.codes[:self.upper_quantizer.total_codes]}
        else:
            return {}

    def update_for_step(self, step, *args):
        self.internal_step = step
        self.upper_quantizer.quantizer.temperature = max(
                    self.upper_quantizer.max_gumbel_temperature * self.upper_quantizer.gumbel_temperature_decay**self.internal_step,
                    self.upper_quantizer.min_gumbel_temperature,
                )


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
    from models.audio.music.transformer_diffusion8 import TransformerDiffusionWithQuantizer
    base_diff = TransformerDiffusionWithQuantizer(in_channels=256, out_channels=512, model_channels=2048, block_channels=1024,
                                                  prenet_channels=1024, prenet_layers=6, num_layers=16, input_vec_dim=1024,
                                                  dropout=.1, unconditioned_percentage=0, freeze_quantizer_until=6000)
    base_diff.load_state_dict(torch.load('x:/dlas/experiments/train_music_diffusion_tfd8/models/47500_generator.pth', map_location=torch.device('cpu')))

    model = GptMusicLower(512, 8, fp16=False)
    model.target_quantizer.load_state_dict(base_diff.quantizer.state_dict(), strict=False)
    torch.save(model.state_dict(), "sample.pth")
    mel = torch.randn(2,256,400)
    model(mel, mel)
    model.get_grad_norm_parameter_groups()


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