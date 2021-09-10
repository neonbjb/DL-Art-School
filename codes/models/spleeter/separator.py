import torch
import torch.nn.functional as F

from models.spleeter.estimator import Estimator


class Separator:
    def __init__(self, model_path, input_sr=44100, device='cuda'):
        self.model = Estimator(2, model_path).to(device)
        self.device = device
        self.input_sr = input_sr

    def separate(self, npwav, normalize=False):
        if not isinstance(npwav, torch.Tensor):
            assert len(npwav.shape) == 1
            wav = torch.tensor(npwav, device=self.device)
            wav = wav.view(1,-1)
        else:
            assert len(npwav.shape) == 2  # Input should be BxL
            wav = npwav.to(self.device)

        if normalize:
            wav = wav / (wav.max() + 1e-8)

        # Spleeter expects audio input to be 44.1kHz.
        wav = F.interpolate(wav.unsqueeze(1), mode='nearest', scale_factor=44100/self.input_sr).squeeze(1)
        res = self.model.separate(wav)
        res = [F.interpolate(r.unsqueeze(1), mode='nearest', scale_factor=self.input_sr/44100)[:,0] for r in res]
        return {
            'vocals': res[0].cpu().numpy(),
            'accompaniment': res[1].cpu().numpy()
        }
