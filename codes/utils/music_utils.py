import torch

from models.audio.mel2vec import ContrastiveTrainingWrapper
from models.audio.music.unet_diffusion_waveform_gen_simple import DiffusionWaveformGen


def get_mel2wav_model():
    model = DiffusionWaveformGen(model_channels=256, in_channels=16, in_mel_channels=256, out_channels=32, channel_mult=[1,2,3,4,4],
                                 num_res_blocks=[3,3,2,2,1], token_conditioning_resolutions=[1,4,16], dropout=0, kernel_size=3, scale_factor=2,
                                 time_embed_dim_multiplier=4, unconditioned_percentage=0)
    model.load_state_dict(torch.load("../experiments/music_mel2wav.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def get_music_codegen():
    model = ContrastiveTrainingWrapper(mel_input_channels=256, inner_dim=1024, layers=24, dropout=0,
                                           mask_time_prob=0,
                                           mask_time_length=6, num_negatives=100, codebook_size=16, codebook_groups=4,
                                           disable_custom_linear_init=True, do_reconstruction_loss=True)
    model.load_state_dict(torch.load(f"../experiments/m2v_music.pth", map_location=torch.device('cpu')))
    model = model.eval()
    return model