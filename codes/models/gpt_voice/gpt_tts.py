import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import munchify
from tqdm import tqdm

from models.arch_util import ConvGnSilu
from models.gpt_voice.pixelshuffle_1d import PixelUnshuffle1D, PixelShuffle1D
from models.tacotron2 import hparams
from models.tacotron2.taco_utils import get_mask_from_lengths
from models.tacotron2.tacotron2 import Postnet
from models.tacotron2.text import symbols
from models.gpt_voice.min_gpt import GPT, GPTConfig
from trainer.networks import register_model


class GptTts(nn.Module):
    def __init__(self):
        super().__init__()
        number_symbols = len(symbols)
        model_dim = 512
        max_symbols_per_phrase = 200
        max_mel_frames = 900
        mel_dim=80

        self.model_dim = model_dim
        self.max_mel_frames = max_mel_frames
        self.text_embedding = nn.Embedding(number_symbols, model_dim)
        # Whenever we process MEL frames, we need to be careful to use casually masked convolutions to avoid adding bias
        # into the model which we cannot provide in inference.
        self.mel_encoder = nn.Sequential(ConvGnSilu(mel_dim, model_dim//2, kernel_size=1, convnd=nn.Conv1d),
                                         PixelUnshuffle1D(2),
                                         ConvGnSilu(model_dim, model_dim, kernel_size=1, convnd=nn.Conv1d),
                                         ConvGnSilu(model_dim, model_dim, kernel_size=1, convnd=nn.Conv1d))
        # *_tags are additively applied to
        self.text_tags = nn.Parameter(torch.randn(1, 1, model_dim)/256.0)
        self.separator = nn.Parameter(torch.randn(1, 1, model_dim))
        self.audio_tags = nn.Parameter(torch.randn(1, 1, model_dim)/256.0)
        self.text_preprocess_xformer = GPT(GPTConfig(max_symbols_per_phrase, n_layer=2, n_head=2, n_embd=model_dim))
        self.gpt = GPT(GPTConfig(1+max_symbols_per_phrase+max_mel_frames//2, n_embd=model_dim, n_head=8))

        self.gate_head = nn.Sequential(ConvGnSilu(model_dim, model_dim, kernel_size=1, convnd=nn.Conv1d),
                                      PixelShuffle1D(2),
                                      ConvGnSilu(model_dim//2, model_dim//2, kernel_size=1, convnd=nn.Conv1d),
                                      ConvGnSilu(model_dim//2, 1, kernel_size=1, norm=False, activation=False, convnd=nn.Conv1d))
        self.mel_head = nn.Sequential(ConvGnSilu(model_dim, model_dim, kernel_size=1, convnd=nn.Conv1d),
                                      PixelShuffle1D(2),
                                      ConvGnSilu(model_dim//2, model_dim//2, kernel_size=1, convnd=nn.Conv1d),
                                      ConvGnSilu(model_dim//2, mel_dim, kernel_size=1, norm=False, activation=False, convnd=nn.Conv1d))
        #self.postnet = Postnet(munchify(hparams.create_hparams()))

    def forward(self, text_inputs, mel_targets, output_lengths):
        # Pad mel_targets to be a multiple of 2
        padded = mel_targets.shape[-1] % 2 != 0
        if padded:
            mel_targets = F.pad(mel_targets, (0,1))

        text_emb = self.text_embedding(text_inputs)
        text_emb = self.text_preprocess_xformer(text_emb, text_emb.shape[1])
        text_emb = text_emb + self.text_tags
        mel_emb = self.mel_encoder(mel_targets).permute(0,2,1)
        mel_emb = mel_emb + self.audio_tags
        emb = torch.cat([text_emb,
                         self.separator.repeat(text_emb.shape[0],1,1),
                         mel_emb], dim=1)
        enc = self.gpt(emb, text_emb.shape[1])
        mel_portion = enc[:, text_emb.shape[1]+1:].permute(0,2,1)
        gates = self.gate_head(mel_portion).squeeze(1)
        mel_pred = self.mel_head(mel_portion)

        # Mask portions of output which we don't need to predict.
        mask = ~get_mask_from_lengths(output_lengths, mel_pred.shape[-1])
        mask = mask.unsqueeze(1).repeat(1, mel_pred.shape[1], 1)
        mel_pred.data.masked_fill_(mask, 0)
        gates.data.masked_fill_(mask[:, 0, :], 1e3)

        if padded:
            mel_pred = mel_pred[:, :, :-1]
            gates = gates[:, :-1]

        #postnet_mel_pred = self.postnet(mel_pred)
        #return mel_pred, postnet_mel_pred, gates
        return mel_pred, gates

    def test_guide(self, mel_guide, amount=50):
        mel_guide = mel_guide[:,:,:amount]
        mel_emb = self.mel_encoder(mel_guide).permute(0,2,1)
        mel_emb = mel_emb + self.audio_tags
        return mel_emb

    def inference(self, text_inputs, mel_guide):
        MEL_HEAD_EXPANSION = 2
        GATE_THRESHOLD = .95

        text_emb = self.text_embedding(text_inputs)
        text_emb = self.text_preprocess_xformer(text_emb, text_emb.shape[1])
        text_emb = text_emb + self.text_tags
        b,s,c = text_emb.shape
        emb = torch.cat([text_emb,
                         self.separator.repeat(text_emb.shape[0],1,1),], dim=1)
                         #self.test_guide(mel_guide)], dim=1)
        completed = torch.zeros((b,), device=text_inputs.device, dtype=torch.bool)
        output = None
        for i in tqdm(range(self.max_mel_frames)):
            enc = self.gpt(emb, text_emb.shape[1])
            inferred = enc[:,s:,:].permute(0,2,1)
            # Create output frames.
            inferred_mel_frame = self.mel_head(inferred)[:,:,-MEL_HEAD_EXPANSION:]
            inferred_mel_frame = inferred_mel_frame * (~completed).float().view(b,1,1)
            if output is None:
                output = inferred_mel_frame
            else:
                output = torch.cat([output, inferred_mel_frame], dim=2)

            # Test termination condition
            gate = F.sigmoid(self.gate_head(inferred)).max(dim=-1).values  # TODO: accept single-frame terminations.
            completed = completed.logical_or((gate > GATE_THRESHOLD).squeeze(1))  # This comprises a latch - but that may not be wise.
            if torch.all(completed):
                break

            # Apply inferred mel_frames to emb for next pass.
            mel_emb = self.mel_encoder(output).permute(0,2,1)
            mel_emb = mel_emb + self.audio_tags
            emb = torch.cat([text_emb,
                             self.separator.repeat(text_emb.shape[0],1,1),
                             mel_emb], dim=1)
            if i == self.max_mel_frames//2:
                print("Warning! Inference hit mel frame cap without encountering a stop token.")
                break

        return output


@register_model
def register_gpt_tts(opt_net, opt):
    return GptTts()


if __name__ == '__main__':
    gpt = GptTts()
    m, g = gpt(torch.randint(high=24, size=(2,60)),
               torch.randn(2,80,747),
               torch.tensor([600,747]))
    print(m.shape)
    #print(p.shape)
    print(g.shape)

    #o = gpt.infer(torch.randint(high=24, size=(2,60)))
    #print(o.shape)


