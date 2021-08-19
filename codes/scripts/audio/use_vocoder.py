import pathlib

import numpy
import torch
from scipy.io import wavfile
from tqdm import tqdm

from models.waveglow.waveglow import WaveGlow


class Vocoder:
    def __init__(self):
        self.model = WaveGlow(n_mel_channels=80, n_flows=12, n_group=8, n_early_size=2, n_early_every=4, WN_config={'n_layers': 8, 'n_channels': 256, 'kernel_size': 3})
        sd = torch.load('../experiments/waveglow_256channels_universal_v5.pth')
        self.model.load_state_dict(sd)
        self.model = self.model.to('cuda')
        self.model.eval()

    def transform_mel_to_audio(self, mel):
        if len(mel.shape) == 2:  # Assume it's missing the batch dimension and fix that.
            mel = mel.unsqueeze(0)
        with torch.no_grad():
            return self.model.infer(mel)


if __name__ == '__main__':
    path = 'data/audio'
    files = list(pathlib.Path(path).glob('*.npy'))

    for inp in tqdm(files):
        inp = str(inp)
        mel = torch.tensor(numpy.load(inp)).to('cuda')
        vocoder = Vocoder()
        wav = vocoder.transform_mel_to_audio(mel)
        wavfile.write(f'{inp}.wav', 22050, wav[0].cpu().numpy())