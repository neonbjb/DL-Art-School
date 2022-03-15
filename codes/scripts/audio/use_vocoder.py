import torch
from scipy.io import wavfile

from models.audio.vocoders.waveglow.waveglow import WaveGlow


class Vocoder:
    def __init__(self):
        self.model = WaveGlow(n_mel_channels=80, n_flows=12, n_group=8, n_early_size=2, n_early_every=4, WN_config={'n_layers': 8, 'n_channels': 256, 'kernel_size': 3})
        sd = torch.load('../experiments/waveglow_256channels_universal_v5.pth')
        self.model.load_state_dict(sd)
        self.model = self.model.cpu()
        self.model.eval()

    def transform_mel_to_audio(self, mel):
        if len(mel.shape) == 2:  # Assume it's missing the batch dimension and fix that.
            mel = mel.unsqueeze(0)
        with torch.no_grad():
            return self.model.infer(mel)


if __name__ == '__main__':
    vocoder = Vocoder()
    m = torch.load('C:\\Users\\jbetk\\Documents\\tmp\\some_audio\\00008.mel').cpu()
    wav = vocoder.transform_mel_to_audio(m)
    wavfile.write(f'0.wav', 22050, wav[0].cpu().numpy())