from copy import deepcopy

from datasets import load_metric

import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

import trainer.eval.evaluator as evaluator
from data import create_dataset, create_dataloader
from models.asr.w2v_wrapper import only_letters, Wav2VecWrapper
from models.tacotron2.text import sequence_to_text

# Librispeech:
# baseline: .045% WER.
# fine-tuned new head (0):  .054% WER
#
# baseline: .328
# 0: .342
# 24000: .346


def tacotron_detokenize(seq):
    return only_letters(sequence_to_text(seq))


fb_processor = None
def fb_detokenize(seq):
    global fb_processor
    if fb_processor is None:
        fb_processor = Wav2Vec2Processor.from_pretrained(f"facebook/wav2vec2-large-960h")
    return fb_processor.decode(seq)


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
                pred_seq = model.inference(clip)
                preds.append(self.detokenizer_fn(pred_seq[0]))
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
                'path': ['y:/bigasr_dataset/mozcv/en/test.tsv'],
                'fetcher_mode': ['mozilla_cv'],
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
    }
    model = Wav2VecWrapper(vocab_size=148, basis_model='facebook/wav2vec2-large-robust-ft-libri-960h', freeze_transformer=True, checkpointing_enabled=False)
    weights = torch.load('X:\\dlas\\experiments/train_wav2vec_mass_diverse_initial_annealing_large_pt/models/7000_wav2vec.pth')
    model.load_state_dict(weights)
    model = model.cuda()
    eval = WerEvaluator(model, opt_eval, env)
    print(eval.perform_eval())