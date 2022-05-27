from copy import deepcopy

#from datasets import load_metric

import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor

import trainer.eval.evaluator as evaluator
from data import create_dataset, create_dataloader
from models.audio.asr.w2v_wrapper import only_letters, Wav2VecWrapper
from models.audio.tts.tacotron2 import sequence_to_text, tacotron_symbols
from pyctcdecode import build_ctcdecoder

# Librispeech:
# baseline: 4.5% WER.
# fine-tuned new head (0):  5.4% WER
# train_wav2vec_mass_large/models/13250_wav2vec.pth: 3.05% WER
# train_wav2vec_mass_large/models/13250_wav2vec.pth with kenlm: 3.34% WER
from utils.util import opt_get


def tacotron_detokenize(seq):
    return only_letters(sequence_to_text(seq))


fb_processor = None
def fb_detokenize(seq):
    global fb_processor
    if fb_processor is None:
        fb_processor = Wav2Vec2Processor.from_pretrained(f"facebook/wav2vec2-large-960h")
    return fb_processor.decode(seq)


def perform_lm_processing(logits, decoder):
    from pyctcdecode.constants import (
        DEFAULT_BEAM_WIDTH,
        DEFAULT_MIN_TOKEN_LOGP,
        DEFAULT_PRUNE_LOGP,
    )

    assert len(logits.shape) == 3 and logits.shape[0] == 1
    decoded_beams = decoder.decode_beams(
        logits[0].cpu().numpy(),
        beam_width=DEFAULT_BEAM_WIDTH,
        beam_prune_logp=DEFAULT_PRUNE_LOGP,
        token_min_logp=DEFAULT_MIN_TOKEN_LOGP
    )
    text = decoded_beams[0][0]
    return only_letters(text.upper())

class WerEvaluator(evaluator.Evaluator):
    """
    Evaluator that produces the WER for a speech recognition model on a test set.
    """
    def __init__(self, model, opt_eval, env, detokenizer_fn=tacotron_detokenize):
        super().__init__(model, opt_eval, env, uses_all_ddp=False)
        self.clip_key = opt_eval['clip_key']
        self.clip_lengths_key = opt_eval['clip_lengths_key']
        self.text_seq_key = opt_eval['text_seq_key']
        self.text_seq_lengths_key = opt_eval['text_seq_lengths_key']
        self.wer_metric = load_metric('wer')
        self.detokenizer_fn = detokenizer_fn

        self.kenlm_model_path = opt_get(opt_eval, ['kenlm_path'], None)
        if self.kenlm_model_path is not None:
            self.kenlm_decoder = build_ctcdecoder(labels=tacotron_symbols(), kenlm_model_path=self.kenlm_model_path)

    def perform_eval(self):
        val_opt = deepcopy(self.env['opt']['datasets']['val'])
        val_opt['batch_size'] = 1  # This is important to ensure no padding.
        val_dataset, collate_fn = create_dataset(val_opt, return_collate=True)
        val_loader = create_dataloader(val_dataset, val_opt, self.env['opt'], None, collate_fn=collate_fn)
        model = self.model.module if hasattr(self.model, 'module') else self.model  # Unwrap DDP models
        model.eval()
        with torch.no_grad():
            preds = []
            reals = []
            for batch in tqdm(val_loader):
                clip = batch[self.clip_key]
                assert clip.shape[0] == 1
                real_seq = batch[self.text_seq_key]
                real_seq_len = batch[self.text_seq_lengths_key][0]
                real_seq = real_seq[:, :real_seq_len]
                real_str = only_letters(sequence_to_text(real_seq[0]))
                if len(real_str) > 0:
                    reals.append(real_str)
                else:
                    continue  # The WER computer doesn't like this scenario.
                clip_len = batch[self.clip_lengths_key][0]
                clip = clip[:, :, :clip_len].cuda()
                logits = model.inference_logits(clip)
                if self.kenlm_model_path is not None:
                    pred = perform_lm_processing(logits, self.kenlm_decoder)
                else:
                    pred_seq = logits.argmax(dim=-1)
                    pred_seq = [model.decode_ctc(p) for p in pred_seq]
                    pred = self.detokenizer_fn(pred_seq[0])
                preds.append(pred)
        wer = self.wer_metric.compute(predictions=preds, references=reals)
        model.train()
        return {'eval_wer': wer}


if __name__ == '__main__':
    env = { 'opt': {
        'datasets': {
            'val': {
                'name': 'mass_test',
                'n_workers': 1,
                'batch_size': 1,
                'mode': 'paired_voice_audio',
                'sample_rate': 16000,
                'path': ['y:/bigasr_dataset/librispeech/test_clean/test_clean.txt'],
                'fetcher_mode': ['libritts'],
                #'path': ['y:/bigasr_dataset/mozcv/en/test.tsv'],
                #'fetcher_mode': ['mozilla_cv'],
                'max_wav_length': 200000,
                'use_bpe_tokenizer': False,
                'max_text_length': 400,
                'load_conditioning': False,
                'phase': 'eval',
            }
        }
    }}
    opt_eval = {
        'clip_key': 'wav',
        'clip_lengths_key': 'wav_lengths',
        'text_seq_key': 'padded_text',
        'text_seq_lengths_key': 'text_lengths',
        #'kenlm_path': 'Y:\\bookscorpus-5gram\\5gram.bin',
    }
    model = Wav2VecWrapper(vocab_size=148, basis_model='facebook/wav2vec2-large-robust-ft-libri-960h', freeze_transformer=True, checkpointing_enabled=False)
    weights = torch.load('D:\\dlas\\experiments\\train_wav2vec_mass_large2\\models\\22500_wav2vec.pth')
    model.load_state_dict(weights)
    model = model.cuda()
    eval = WerEvaluator(model, opt_eval, env)
    print(eval.perform_eval())