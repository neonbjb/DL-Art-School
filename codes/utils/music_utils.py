import torch


def get_mel2wav_model():
    from models.audio.music.unet_diffusion_waveform_gen_simple import DiffusionWaveformGen
    model = DiffusionWaveformGen(model_channels=256, in_channels=16, in_mel_channels=256, out_channels=32, channel_mult=[1,2,3,4,4],
                                 num_res_blocks=[3,3,2,2,1], token_conditioning_resolutions=[1,4,16], dropout=0, kernel_size=3, scale_factor=2,
                                 time_embed_dim_multiplier=4, unconditioned_percentage=0)
    model.load_state_dict(torch.load("../experiments/music_mel2wav.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


def get_mel2wav_v3_model():
    from models.audio.music.unet_diffusion_waveform_gen3 import DiffusionWaveformGen
    model = DiffusionWaveformGen(model_channels=256, in_channels=16, in_mel_channels=256, out_channels=32, channel_mult=[1,1.5,2,4],
                                 num_res_blocks=[2,1,1,0], mid_resnet_depth=24, token_conditioning_resolutions=[1,4],
                                 dropout=0, time_embed_dim_multiplier=1, unconditioned_percentage=0)
    model.load_state_dict(torch.load("../experiments/music_mel2wav_v3.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


def get_music_codegen():
    from models.audio.mel2vec import ContrastiveTrainingWrapper
    model = ContrastiveTrainingWrapper(mel_input_channels=256, inner_dim=1024, layers=24, dropout=0,
                                           mask_time_prob=0,
                                           mask_time_length=6, num_negatives=100, codebook_size=16, codebook_groups=4,
                                           disable_custom_linear_init=True, do_reconstruction_loss=True)
    model.load_state_dict(torch.load(f"../experiments/m2v_music.pth", map_location=torch.device('cpu')))
    model = model.eval()
    return model


def get_cheater_encoder():
    from models.audio.music.gpt_music2 import UpperEncoder
    encoder = UpperEncoder(256, 1024, 256)
    encoder.load_state_dict(
        torch.load('../experiments/music_cheater_encoder_256.pth', map_location=torch.device('cpu')))
    encoder = encoder.eval()
    return encoder


def get_cheater_decoder():
    from models.audio.music.transformer_diffusion12 import TransformerDiffusionWithCheaterLatent
    model = TransformerDiffusionWithCheaterLatent(in_channels=256, out_channels=512, model_channels=1024,
                                                         contraction_dim=512, prenet_channels=1024, input_vec_dim=256,
                                                         prenet_layers=6, num_heads=8, num_layers=16, new_code_expansion=True,
                                                         dropout=0, unconditioned_percentage=0)
    model.load_state_dict(torch.load(f'../experiments/music_cheater_decoder.pth', map_location=torch.device('cpu')))
    model = model.eval()
    return model


def get_ar_prior():
    from models.audio.music.cheater_gen_ar import ConditioningAR
    cheater_ar = ConditioningAR(1024, layers=24, dropout=0, cond_free_percent=0)
    cheater_ar.load_state_dict(torch.load('../experiments/music_cheater_ar.pth', map_location=torch.device('cpu')))
    #cheater_ar = cheater_ar.eval()
    return cheater_ar