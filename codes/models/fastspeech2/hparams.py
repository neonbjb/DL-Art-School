import os

from munch import munchify

def get_hparams():
    return munchify({
        'mfa_path': "./MFA",
        'dataset': 'LJSpeech',

        ### Text ###
        # g2p_en
        'text_cleaners': ['english_cleaners'],

        ### FastSpeech 2 ###
        'encoder_layer': 4,
        'encoder_head': 2,
        'encoder_hidden': 256,

        'decoder_layer': 4,
        'decoder_head': 2,
        'decoder_hidden': 256,

        'fft_conv1d_filter_size': 1024,
        'fft_conv1d_kernel_size': (9, 1),

        'encoder_dropout': 0.2,
        'decoder_dropout': 0.2,

        'variance_predictor_filter_size': 256,
        'variance_predictor_kernel_size': 3,
        'variance_predictor_dropout': 0.5,

        'max_seq_len': 1000,

        'max_wav_value': 32768.0,
        'n_mel_channels': 80,
        'mel_fmin': 0.0,
        'mel_fmax': None,

        # Audio and mel
        'sampling_rate': 22050,
        'filter_length': 800,
        'hop_length': 200,
        'win_length': 800,

        # Quantization for F0 and energy
        'f0_min': 71,
        'f0_max': 786.7,
        'energy_min': 21.23,
        'energy_max': 101.02,
        'n_bins': 256,

        # Speaker embedding
        'use_spk_embed': False,
        'spk_embed_dim': 256,
        'spk_embed_weight_std': 0.01,

        # Log-scaled duration
        'log_offset': 1.,
    })

defaults = get_hparams()
