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
    NUMBER_SYMBOLS = len(symbols)+3
    TEXT_START_TOKEN = NUMBER_SYMBOLS-3
    TEXT_STOP_TOKEN = NUMBER_SYMBOLS-2
    TEXT_PAD_TOKEN = NUMBER_SYMBOLS-1
    MEL_DICTIONARY_SIZE = 512+3
    MEL_START_TOKEN = MEL_DICTIONARY_SIZE-3
    MEL_STOP_TOKEN = MEL_DICTIONARY_SIZE-2
    MEL_PAD_TOKEN = MEL_DICTIONARY_SIZE-1

    def __init__(self):
        super().__init__()
        model_dim = 512
        max_symbols_per_phrase = 200
        max_mel_frames = 900 * 3 // 8  #  The VQVAE outputs 3/8 of the input mel as tokens.
        mel_dim=80

        self.model_dim = model_dim
        self.max_mel_frames = max_mel_frames
        self.text_embedding = nn.Embedding(self.NUMBER_SYMBOLS, model_dim)
        self.mel_embedding = nn.Embedding(self.MEL_DICTIONARY_SIZE, model_dim)
        # *_tags are additively applied to
        self.text_pos_embedding = nn.Embedding(max_symbols_per_phrase, model_dim)
        self.mel_pos_embedding = nn.Embedding(max_mel_frames, model_dim)
        self.gpt = GPT(GPTConfig(1+max_symbols_per_phrase+max_mel_frames, n_embd=model_dim, n_head=8), do_pos_emb=False)

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.NUMBER_SYMBOLS)
        self.mel_head = nn.Linear(model_dim, self.MEL_DICTIONARY_SIZE)

    def forward(self, text_inputs, text_lengths, mel_targets, output_lengths):
        output_lengths = output_lengths * 3 // 8  # The data we are dealing with has been compressed by the vqvae.
        # Add the stop tokens to the end of the texts and mels. Theoretically this would be better done at the dataloader level.
        batch_range = torch.arange(0, text_inputs.shape[0])
        text_inputs = F.pad(text_inputs, (0,1))
        text_inputs.index_put_((batch_range, text_lengths), torch.tensor([self.TEXT_STOP_TOKEN], dtype=torch.long, device=text_inputs.device))
        text_lengths = text_lengths + 1
        mel_targets = F.pad(mel_targets, (0,1))
        mel_targets.index_put_((batch_range, output_lengths), torch.tensor([self.MEL_STOP_TOKEN], dtype=torch.long, device=text_inputs.device))
        output_lengths = output_lengths + 1
        # Add the start tokens to the beginnings of the texts and mels.
        text_inputs = F.pad(text_inputs, (1,0), value=self.TEXT_START_TOKEN)
        text_lengths = text_lengths + 1
        mel_targets = F.pad(mel_targets, (1,0), value=self.MEL_START_TOKEN)
        output_lengths = output_lengths + 1
        # Add padding as well. This also should realistically be done at the dataloader level.
        text_pad_mask = ~get_mask_from_lengths(text_lengths, text_inputs.shape[1])
        text_inputs.data.masked_fill_(text_pad_mask, self.TEXT_PAD_TOKEN)
        mel_pad_mask = ~get_mask_from_lengths(output_lengths, mel_targets.shape[1])
        mel_targets.data.masked_fill_(mel_pad_mask, self.MEL_PAD_TOKEN)

        text_emb = self.text_embedding(text_inputs)
        text_emb = text_emb + self.text_pos_embedding(torch.arange(text_inputs.shape[1], device=text_inputs.device))
        mel_emb = self.mel_embedding(mel_targets)
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_targets.shape[1], device=mel_targets.device))
        emb = torch.cat([text_emb, mel_emb], dim=1)
        enc = self.gpt(emb)

        # Compute logits for text and mel heads
        text_logits = self.final_norm(enc[:, :text_emb.shape[1]])
        text_logits = self.text_head(text_logits)
        mel_logits = self.final_norm(enc[:, text_emb.shape[1]:])
        mel_logits = self.mel_head(mel_logits)

        # Compute loss
        loss_text = F.cross_entropy(text_logits.permute(0,2,1)[:,:,1:], text_inputs[:,1:], reduction='none')
        loss_mel = F.cross_entropy(mel_logits.permute(0,2,1)[:,:,1:], mel_targets[:,1:], reduction='none')
        # Apply a reduction factor across MEL_PAD and TEXT_PAD tokens.
        pad_loss_reduction_factor = .01
        loss_text = loss_text * torch.ones_like(loss_text).masked_fill_(text_pad_mask[:,1:], pad_loss_reduction_factor)
        loss_mel = loss_mel * torch.ones_like(loss_mel).masked_fill_(mel_pad_mask[:,1:], pad_loss_reduction_factor)

        # Fix up mel_logits so it can go into a VAE decoder as well.
        mel_codes = torch.argmax(F.softmax(mel_logits, dim=-1), dim=-1)
        mel_codes = mel_codes[:,1:]
        mel_codes = mel_codes * torch.ones_like(mel_codes).masked_fill_(mel_pad_mask[:,1:], 0)
        mel_codes = mel_codes[:,:-1]
        extra_mask = mel_codes < self.MEL_DICTIONARY_SIZE-3  # The VAE doesn't know about START/STOP/PAD
        mel_codes = mel_codes * extra_mask

        return loss_text.mean(), loss_mel.mean(), mel_codes

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
    l1, l2, i = gpt(torch.randint(high=24, size=(2,60)),
               torch.tensor([55,58]),
               torch.randint(high=512, size=(2,310)),
               torch.tensor([300,305]))
    print(i.shape)

    #o = gpt.infer(torch.randint(high=24, size=(2,60)))
    #print(o.shape)


