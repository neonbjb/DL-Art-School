import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from models.audio.music.music_quantizer import MusicQuantizer
from models.audio.music.music_quantizer2 import MusicQuantizer2
from trainer.networks import register_model
from utils.util import opt_get


class GptMusic(nn.Module):
    def __init__(self, dim, layers, num_target_vectors=512, num_target_groups=2, cv_dim=1024, num_upper_vectors=64, num_upper_groups=4):
        super().__init__()
        self.num_groups = num_target_groups
        self.config = GPT2Config(vocab_size=1, n_positions=8192, n_embd=dim, n_layer=layers, n_head=dim//64,
                                 n_inner=dim*2)
        self.target_quantizer = MusicQuantizer(inp_channels=256, inner_dim=[1024,1024,512], codevector_dim=cv_dim, codebook_size=num_target_vectors, codebook_groups=num_target_groups)
        del self.target_quantizer.decoder
        del self.target_quantizer.up
        self.upper_quantizer = MusicQuantizer2(inp_channels=256, inner_dim=[1024,896,768,640,512,384], codevector_dim=cv_dim, codebook_size=num_upper_vectors, codebook_groups=num_upper_groups)
        del self.upper_quantizer.up
        self.gpt = GPT2Model(self.config)
        del self.gpt.wte  # Unused, we'll do our own embeddings.
        self.embeddings = nn.ModuleList([nn.Embedding(num_target_vectors, dim // num_target_groups) for _ in range(num_target_groups)])
        self.upper_proj = nn.Conv1d(cv_dim, dim, kernel_size=1)
        self.heads = nn.ModuleList([nn.Linear(dim, num_target_vectors) for _ in range(num_target_groups)])


    def forward(self, mel):
        with torch.no_grad():
            self.target_quantizer.eval()
            codes = self.target_quantizer.get_codes(mel)
        upper_vector, upper_diversity = self.upper_quantizer(mel, return_decoder_latent=True)
        upper_vector = self.upper_proj(upper_vector)
        upper_vector = F.interpolate(upper_vector, size=codes.shape[1], mode='linear')
        upper_vector = upper_vector.permute(0,2,1)

        inputs = codes[:, :-1]
        upper_vector = upper_vector[:, :-1]
        targets = codes[:, 1:]

        h = [embedding(inputs[:, :, i]) for i, embedding in enumerate(self.embeddings)]
        h = torch.cat(h, dim=-1) + upper_vector
        h = self.gpt(inputs_embeds=h, return_dict=True).last_hidden_state

        losses = 0
        for i, head in enumerate(self.heads):
            logits = head(h).permute(0,2,1)
            loss = F.cross_entropy(logits, targets[:,:,i])
            losses = losses + loss

        return losses / self.num_groups


@register_model
def register_music_gpt(opt_net, opt):
    return GptMusic(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    model = GptMusic(512, 12)
    mel = torch.randn(2,256,400)
    model(mel)