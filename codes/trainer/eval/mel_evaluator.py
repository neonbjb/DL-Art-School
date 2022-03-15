import torch

import trainer.eval.evaluator as evaluator

from data import create_dataset
from data.audio.nv_tacotron_dataset import TextMelCollate
from models.audio.tts.tacotron2 import Tacotron2LossRaw
from torch.utils.data import DataLoader
from tqdm import tqdm


# Evaluates the performance of a MEL spectrogram predictor.
class MelEvaluator(evaluator.Evaluator):
    def __init__(self, model, opt_eval, env):
        super().__init__(model, opt_eval, env, uses_all_ddp=True)
        self.batch_sz = opt_eval['batch_size']
        self.dataset = create_dataset(opt_eval['dataset'])
        assert self.batch_sz is not None
        self.dataloader = DataLoader(self.dataset, self.batch_sz, shuffle=False, num_workers=1, collate_fn=TextMelCollate(n_frames_per_step=1))
        self.criterion = Tacotron2LossRaw()

    def perform_eval(self):
        counter = 0
        total_error = 0
        self.model.eval()
        for batch in tqdm(self.dataloader):
            model_params = {
                'text_inputs': 'padded_text',
                'text_lengths': 'input_lengths',
                'mels': 'padded_mel',
                'output_lengths': 'output_lengths',
            }
            params = {k: batch[v].to(self.env['device']) for k, v in model_params.items()}
            with torch.no_grad():
                pred = self.model(**params)

            targets = ['padded_mel', 'padded_gate']
            targets = [batch[t].to(self.env['device']) for t in targets]
            total_error += self.criterion(pred, targets).item()
            counter += 1
        self.model.train()

        return {"validation-score": total_error / counter}

