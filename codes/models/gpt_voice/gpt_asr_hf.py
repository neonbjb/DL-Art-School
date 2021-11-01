import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config

from models.tacotron2.text import symbols
from trainer.networks import register_model
from utils.util import opt_get


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.BatchNorm1d(chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.BatchNorm1d(chan)
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class MelEncoder(nn.Module):
    def __init__(self, channels, mel_channels=80):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(nn.Conv1d(mel_channels, channels//4, kernel_size=5, padding=2),
                                     ResBlock(channels//4),
                                     ResBlock(channels//4),
                                     nn.Conv1d(channels//4, channels//2, kernel_size=3, stride=2, padding=1),
                                     nn.BatchNorm1d(channels//2),
                                     nn.ReLU(),
                                     ResBlock(channels//2),
                                     ResBlock(channels//2),
                                     nn.Conv1d(channels//2, channels, kernel_size=3, stride=2, padding=1),
                                     ResBlock(channels),
                                     ResBlock(channels)
                                     )

    def forward(self, x):
        return self.encoder(x)


class GptAsrHf(nn.Module):
    NUMBER_SYMBOLS = len(symbols)
    NUMBER_TEXT_TOKENS = NUMBER_SYMBOLS+1

    def __init__(self, layers=8, model_dim=512, heads=8, max_symbols_per_phrase=200, max_mel_frames=1000):
        super().__init__()
        self.max_mel_frames = max_mel_frames // 4  # Mel frames are reduced by a factor of 4 during encoding.
        self.max_symbols_per_phrase = max_symbols_per_phrase

        self.model_dim = model_dim
        self.max_mel_frames = self.max_mel_frames
        self.mel_encoder = MelEncoder(model_dim)
        self.text_pos_embedding = nn.Embedding(self.max_symbols_per_phrase + 1, model_dim)
        self.mel_pos_embedding = nn.Embedding(self.max_mel_frames, model_dim)
        seq_length = 2+self.max_symbols_per_phrase+self.max_mel_frames
        self.gpt = GPT2Model(GPT2Config(vocab_size=self.NUMBER_TEXT_TOKENS,
                                        n_positions=seq_length,
                                        n_ctx=seq_length,
                                        n_embd=model_dim,
                                        n_layer=layers,
                                        n_head=heads))
        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.NUMBER_TEXT_TOKENS)

    def get_logits(self, mel_inputs, text_targets):
        # Pad front and back. Pad at front is the "START" token.
        text_targets = F.pad(text_targets, (1,0), value=self.NUMBER_SYMBOLS)
        text_targets = F.pad(text_targets, (0, self.max_symbols_per_phrase - text_targets.shape[1]))
        text_emb = self.gpt.get_input_embeddings()(text_targets)
        text_emb = text_emb + self.text_pos_embedding(torch.arange(text_emb.shape[1], device=text_targets.device))
        mel_emb = self.mel_encoder(mel_inputs)
        mel_emb = F.pad(mel_emb, (0, self.max_mel_frames - mel_emb.shape[-1]))
        mel_emb = mel_emb.permute(0,2,1).contiguous()
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_emb.device))
        emb = torch.cat([mel_emb, text_emb], dim=1)
        enc = self.gpt(inputs_embeds=emb, return_dict=True).last_hidden_state
        text_logits = self.final_norm(enc[:, self.max_mel_frames:])
        text_logits = self.text_head(text_logits)
        text_logits = text_logits.permute(0,2,1)
        return text_logits

    def forward(self, mel_inputs, text_targets):
        text_logits = self.get_logits(mel_inputs, text_targets)
        loss_text = F.cross_entropy(text_logits[:,:,:-1], text_targets[:,1:].long())
        return loss_text.mean(), text_logits


@register_model
def register_gpt_asr_hf(opt_net, opt):
    return GptAsrHf(**opt_get(opt_net, ['kwargs'], {}))


# Quick script that loads a model and halves the number of layers, then saves that model.
def distill():
    gpt = GptAsrHf(max_symbols_per_phrase=250, max_mel_frames=1400, layers=12, model_dim=768, heads=12)
    gpt.load_state_dict(torch.load('../experiments/train_gpt_asr_mass/models/21500_mel_gen.pth'))
    rc = 0
    i = 0
    while i < len(gpt.gpt.layers.layers):
        if rc % 2 != 0:
            del gpt.gpt.layers.layers[i]
        else:
            i += 1
        rc += 1
    torch.save(gpt.state_dict(), '../experiments/train_gpt_asr_mass/models/21500_mel_gen_distilled.pth')


if __name__ == '__main__':
    gpt = GptAsrHf(max_symbols_per_phrase=100, max_mel_frames=200, layers=6, model_dim=256, heads=2)
    l = gpt(torch.randn(2,80,800), torch.randint(high=len(symbols), size=(2,100)))

    '''
    with torch.no_grad():
        t = torch.randn(1,80,800).cuda()
        start = time()
        s = gpt.inference_beam_topk(t)
        print(time()-start)

        start = time()
        o = gpt.inference_beam_topk(t, fn='inference_beam_opt')
        print(time()-start)
    '''

