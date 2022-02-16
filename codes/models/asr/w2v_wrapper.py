from itertools import groupby

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

from data.audio.unsupervised_audio_dataset import load_audio
from models.tacotron2.text import symbols, sequence_to_text
from trainer.networks import register_model
from utils.util import opt_get


def only_letters(string):
    allowlist = set(' ABCDEFGHIJKLMNOPQRSTUVWXYZ\'')
    return ''.join(filter(allowlist.__contains__, string.upper()))


class Wav2VecWrapper(nn.Module):
    """
    Basic wrapper class that makes Wav2Vec2 usable by DLAS.
    """
    def __init__(self, vocab_size=148, basis_model='facebook/wav2vec2-large', freeze_transformer=False, output_wer=True, checkpointing_enabled=True, provide_attention_mask=False):
        super().__init__()
        self.provide_attention_mask = provide_attention_mask
        
        self.w2v = Wav2Vec2ForCTC.from_pretrained(basis_model)
        # Perform some surgery to get the model we actually want.
        self.w2v.wav2vec2.encoder.gradient_checkpointing = checkpointing_enabled
        self.w2v.lm_head = nn.Linear(self.w2v.config.hidden_size, vocab_size)
        self.w2v.config.vocab_size = vocab_size
        self.w2v.config.pad_token_id = 0
        self.w2v.config.ctc_loss_reduction = 'mean'
        self.w2v.config.apply_spec_augment = True

        # We always freeze the feature extractor, which needs some special operations in DLAS
        for p in self.w2v.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False
            p.DO_NOT_TRAIN = True
        if freeze_transformer:
            # Also freeze the encoder here.
            for p in list(self.w2v.wav2vec2.encoder.parameters()) + list(self.w2v.wav2vec2.feature_projection.parameters()):
                p.requires_grad = False
                p.DO_NOT_TRAIN = True

        self.output_wer = output_wer
        if output_wer:
            self.last_pred = []
            self.last_labels = []

    def forward(self, audio, unaligned_tokens, wav_lengths, text_lengths):
        audio = audio[:, :, :wav_lengths.max()]
        unaligned_tokens = unaligned_tokens[:, :text_lengths.max()]
        attention_mask = torch.ones_like(audio).squeeze(1)
        for b in range(audio.shape[0]):
            attention_mask[b, wav_lengths[b]:] = 0
            unaligned_tokens[b, text_lengths[b]:] = -100

        audio_norm = (audio - audio.mean()) / torch.sqrt(audio.var() + 1e-7)
        if self.provide_attention_mask:
            outputs = self.w2v(input_values=audio_norm.squeeze(1), attention_mask=attention_mask, labels=unaligned_tokens)
        else:
            outputs = self.w2v(input_values=audio_norm.squeeze(1), labels=unaligned_tokens)

        if self.output_wer:
            self.last_pred.append(torch.argmax(outputs.logits, dim=-1))
            if len(self.last_pred) > 10:
                self.last_pred = self.last_pred[1:]
            self.last_labels.append(unaligned_tokens)
            if len(self.last_labels) > 10:
                self.last_labels = self.last_labels[1:]
        return outputs.loss

    def decode_ctc(self, output):
        if isinstance(output, torch.Tensor):
            output = output.tolist()
        tokens = [token_group[0] for token_group in groupby(output)]
        filtered_tokens = list(filter(lambda token: token != 0, tokens))
        return filtered_tokens

    def get_debug_values(self, step, net_name):
        res = {}
        if self.output_wer and step % 100 == 0:
            from datasets import load_metric
            wer_metric = load_metric("wer")
            label_strings = []
            pred_strings = []
            for last_labels, last_pred in zip(self.last_labels, self.last_pred):
                last_labels[last_labels == -100] = 0
                label_strings.extend([only_letters(sequence_to_text(lbl)) for lbl in last_labels])
                pred_strings.extend([only_letters(sequence_to_text(self.decode_ctc(pred))) for pred in last_pred])
            wer = wer_metric.compute(predictions=pred_strings, references=label_strings)
            res['wer'] = wer
            print(f"Sample prediction: {pred_strings[0]} <=> {label_strings[0]}")
        return res

    def inference(self, audio):
        audio_norm = (audio - audio.mean()) / torch.sqrt(audio.var() + 1e-7)
        logits = self.w2v(input_values=audio_norm.squeeze(1)).logits
        pred = logits.argmax(dim=-1)
        return [self.decode_ctc(p) for p in pred]


@register_model
def register_wav2vec2_finetune(opt_net, opt):
    return Wav2VecWrapper(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    print(only_letters("Hello, world!"))
    w2v = Wav2VecWrapper(basis_model='facebook/wav2vec2-large-960h', freeze_transformer=True)
    loss = w2v(torch.randn(2,1,50000), torch.randint(0,40,(2,70)), torch.tensor([20000, 30000]), torch.tensor([35, 50]))
    w2v.get_debug_values(0,"")

    sd = torch.load('../experiments/train_wav2vec_mass_archived_r0/models/19500_wav2vec.pth')
    w2v.load_state_dict(sd)
    pred = w2v.inference(load_audio('Y:\\clips\\books1\\754_Dan Simmons - The Rise Of Endymion 356 of 450\\00026.wav', 16000).unsqueeze(0))
    res = sequence_to_text(pred[0])
    print(res)
