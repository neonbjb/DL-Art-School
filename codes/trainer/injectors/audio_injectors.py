import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from trainer.inject import Injector
from utils.util import opt_get


class MelSpectrogramInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        from models.tacotron2.layers import TacotronSTFT
        # These are the default tacotron values for the MEL spectrogram.
        filter_length = opt_get(opt, ['filter_length'], 1024)
        hop_length = opt_get(opt, ['hop_length'], 256)
        win_length = opt_get(opt, ['win_length'], 1024)
        n_mel_channels = opt_get(opt, ['n_mel_channels'], 80)
        mel_fmin = opt_get(opt, ['mel_fmin'], 0)
        mel_fmax = opt_get(opt, ['mel_fmax'], 8000)
        sampling_rate = opt_get(opt, ['sampling_rate'], 22050)
        self.stft = TacotronSTFT(filter_length, hop_length, win_length, n_mel_channels, sampling_rate, mel_fmin, mel_fmax)

    def forward(self, state):
        inp = state[self.input]
        if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        self.stft = self.stft.to(inp.device)
        return {self.output: self.stft.mel_spectrogram(inp)}


class TorchMelSpectrogramInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = opt_get(opt, ['filter_length'], 1024)
        self.hop_length = opt_get(opt, ['hop_length'], 256)
        self.win_length = opt_get(opt, ['win_length'], 1024)
        self.n_mel_channels = opt_get(opt, ['n_mel_channels'], 80)
        self.mel_fmin = opt_get(opt, ['mel_fmin'], 0)
        self.mel_fmax = opt_get(opt, ['mel_fmax'], 8000)
        self.sampling_rate = opt_get(opt, ['sampling_rate'], 22050)
        norm = opt_get(opt, ['normalize'], False)
        self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=self.filter_length, hop_length=self.hop_length,
                                                             win_length=self.win_length, power=2, normalized=norm,
                                                             sample_rate=self.sampling_rate, f_min=self.mel_fmin,
                                                             f_max=self.mel_fmax, n_mels=self.n_mel_channels,
                                                             norm="slaney")
        self.mel_norm_file = opt_get(opt, ['mel_norm_file'], None)
        if self.mel_norm_file is not None:
            self.mel_norms = torch.load(self.mel_norm_file)
        else:
            self.mel_norms = None

    def forward(self, state):
        inp = state[self.input]
        if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        self.mel_stft = self.mel_stft.to(inp.device)
        mel = self.mel_stft(inp)
        # Perform dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if self.mel_norms is not None:
            self.mel_norms = self.mel_norms.to(mel.device)
            mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
        return {self.output: mel}


class RandomAudioCropInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.crop_sz = opt['crop_size']

    def forward(self, state):
        inp = state[self.input]
        len = inp.shape[-1]
        margin = len - self.crop_sz
        start = random.randint(0, margin)
        return {self.output: inp[:, :, start:start+self.crop_sz]}


class AudioClipInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.clip_size = opt['clip_size']
        self.ctc_codes = opt['ctc_codes_key']
        self.output_ctc = opt['ctc_out_key']

    def forward(self, state):
        inp = state[self.input]
        ctc = state[self.ctc_codes]
        len = inp.shape[-1]
        if len > self.clip_size:
            proportion_inp_remaining = self.clip_size/len
            inp = inp[:, :, :self.clip_size]
            ctc = ctc[:,:int(proportion_inp_remaining*ctc.shape[-1])]
        return {self.output: inp, self.output_ctc: ctc}


class AudioResampleInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.input_sr = opt['input_sample_rate']
        self.output_sr = opt['output_sample_rate']

    def forward(self, state):
        inp = state[self.input]
        return {self.output: torchaudio.functional.resample(inp, self.input_sr, self.output_sr)}
