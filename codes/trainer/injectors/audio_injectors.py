import random

import torch
import torch.nn.functional as F
import torchaudio

from trainer.inject import Injector
from utils.util import opt_get, load_model_from_config

TACOTRON_MEL_MAX = 2.3143386840820312
TACOTRON_MEL_MIN = -11.512925148010254

def normalize_tacotron_mel(mel):
    return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1

def denormalize_tacotron_mel(norm_mel):
    return ((norm_mel+1)/2)*(TACOTRON_MEL_MAX-TACOTRON_MEL_MIN)+TACOTRON_MEL_MIN

class MelSpectrogramInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        from models.audio.tts.tacotron2 import TacotronSTFT
        # These are the default tacotron values for the MEL spectrogram.
        filter_length = opt_get(opt, ['filter_length'], 1024)
        hop_length = opt_get(opt, ['hop_length'], 256)
        win_length = opt_get(opt, ['win_length'], 1024)
        n_mel_channels = opt_get(opt, ['n_mel_channels'], 80)
        mel_fmin = opt_get(opt, ['mel_fmin'], 0)
        mel_fmax = opt_get(opt, ['mel_fmax'], 8000)
        sampling_rate = opt_get(opt, ['sampling_rate'], 22050)
        self.stft = TacotronSTFT(filter_length, hop_length, win_length, n_mel_channels, sampling_rate, mel_fmin, mel_fmax)
        self.do_normalization = opt_get(opt, ['do_normalization'], None)  # This is different from the TorchMelSpectrogramInjector. This just normalizes to the range [-1,1]

    def forward(self, state):
        inp = state[self.input]
        if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        self.stft = self.stft.to(inp.device)
        mel = self.stft.mel_spectrogram(inp)
        if self.do_normalization:
            mel = normalize_tacotron_mel(mel)
        return {self.output: mel}


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
        self.lengths_key = opt['lengths_key']

    def forward(self, state):
        inp = state[self.input]
        lens = state[self.lengths_key]
        len = torch.min(lens)
        margin = len - self.crop_sz
        if margin < 0:
            return {self.output: inp}
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


class DiscreteTokenInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        cfg = opt_get(opt, ['dvae_config'], "../experiments/train_diffusion_vocoder_22k_level.yml")
        dvae_name = opt_get(opt, ['dvae_name'], 'dvae')
        self.dvae = load_model_from_config(cfg, dvae_name, device=f'cuda:{env["device"]}').eval()

    def forward(self, state):
        inp = state[self.input]
        with torch.no_grad():
            self.dvae = self.dvae.to(inp.device)
            codes = self.dvae.get_codebook_indices(inp)
            return {self.output: codes}


class GptVoiceLatentInjector(Injector):
    """
    This injector does all the legwork to generate latents out of a UnifiedVoice model, including encoding all audio
    inputs into a MEL spectrogram and discretizing the inputs.
    """
    def __init__(self, opt, env):
        super().__init__(opt, env)
        # For discrete tokenization.
        cfg = opt_get(opt, ['dvae_config'], "../experiments/train_diffusion_vocoder_22k_level.yml")
        dvae_name = opt_get(opt, ['dvae_name'], 'dvae')
        self.dvae = load_model_from_config(cfg, dvae_name).cuda().eval()
        # The unified_voice model.
        cfg = opt_get(opt, ['gpt_config'], "../experiments/train_gpt_tts_unified.yml")
        model_name = opt_get(opt, ['gpt_name'], 'gpt')
        pretrained_path = opt['gpt_path']
        self.gpt = load_model_from_config(cfg, model_name=model_name,
                                          also_load_savepoint=False, load_path=pretrained_path).cuda().eval()
        # Mel converter
        self.mel_inj = TorchMelSpectrogramInjector({'in': 'wav', 'out': 'mel', 'mel_norm_file': '../experiments/clips_mel_norms.pth'},{})
        # Aux input keys.
        self.conditioning_key = opt['conditioning_clip']
        self.text_input_key = opt['text']
        self.text_lengths_key = opt['text_lengths']
        self.input_lengths_key = opt['input_lengths']

    def to_mel(self, t):
        return self.mel_inj({'wav': t})['mel']

    def forward(self, state):
        with torch.no_grad():
            mel_inputs = self.to_mel(state[self.input])
            mel_cond = self.to_mel(state[self.conditioning_key])

            # Use the input as a conditioning input as well. This is fine because we are not actually training the GPT network so it can't learn to cheat.
            max_mel_len = max(mel_inputs.shape[-1], mel_cond.shape[-1])
            mel_cond = F.pad(mel_cond, (0, max_mel_len-mel_cond.shape[-1]))
            mel_cond2 = F.pad(mel_inputs, (0, max_mel_len-mel_inputs.shape[-1]))
            mel_cond = torch.cat([mel_cond.unsqueeze(1), mel_cond2.unsqueeze(1)], dim=1)
            self.dvae = self.dvae.to(mel_inputs.device)
            codes = self.dvae.get_codebook_indices(mel_inputs)
            self.gpt = self.gpt.to(codes.device)
            latents = self.gpt.forward(mel_cond, state[self.text_input_key],
                                       state[self.text_lengths_key], codes, state[self.input_lengths_key],
                                       text_first=True, raw_mels=None, return_attentions=False, return_latent=True)
            return {self.output: latents}
