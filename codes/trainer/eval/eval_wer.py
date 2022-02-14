from copy import deepcopy

from datasets import load_metric

import torch
import trainer.eval.evaluator as evaluator
from data import create_dataset, create_dataloader
from models.asr.w2v_wrapper import only_letters
from models.tacotron2.text import sequence_to_text


class WerEvaluator(evaluator.Evaluator):
    """
    Evaluator that produces the WER for a speech recognition model on a test set.
    """
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env, uses_all_ddp=False)
        self.clip_key = opt_eval['clip_key']
        self.clip_lengths_key = opt_eval['clip_lengths_key']
        self.text_seq_key = opt_eval['text_seq_key']
        self.text_seq_lengths_key = opt_eval['text_seq_lengths_key']
        self.wer_metric = load_metric('wer')

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
            for batch in val_loader:
                clip = batch[self.clip_key]
                assert clip.shape[0] == 1
                clip_len = batch[self.clip_lengths_key][0]
                clip = clip[:, :, :clip_len].cuda()
                pred_seq = model.inference(clip)
                preds.append(only_letters(sequence_to_text(pred_seq[0])))
                real_seq = batch[self.text_seq_key]
                real_seq_len = batch[self.text_seq_lengths_key][0]
                real_seq = real_seq[:, :real_seq_len]
                reals.append(only_letters(sequence_to_text(real_seq[0])))
        wer = self.wer_metric.compute(predictions=preds, references=reals)
        model.train()
        return {'eval_wer': wer}

