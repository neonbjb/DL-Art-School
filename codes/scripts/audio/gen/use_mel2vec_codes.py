import torch
import torchvision

from models.audio.mel2vec import ContrastiveTrainingWrapper
from trainer.injectors.audio_injectors import TorchMelSpectrogramInjector, normalize_mel
from utils.util import load_audio

def collapse_codegroups(codes):
    codes = codes.clone()
    groups = codes.shape[-1]
    for k in range(groups):
        codes[:,:,k] = codes[:,:,k] * groups ** k
    codes = codes.sum(-1)
    return codes


def recover_codegroups(codes, groups):
    codes = codes.clone()
    output = torch.LongTensor(codes.shape[0], codes.shape[1], groups, device=codes.device)
    for k in range(groups):
        output[:,:,k] = codes % groups
        codes = codes // groups
    return output


if __name__ == '__main__':
    model = ContrastiveTrainingWrapper(mel_input_channels=256, inner_dim=1024, layers=24, dropout=0, mask_time_prob=0,
                                       mask_time_length=6, num_negatives=100, codebook_size=16, codebook_groups=4,
                                       disable_custom_linear_init=True, feature_producer_type='standard',
                                       freq_mask_percent=0, do_reconstruction_loss=True)
    model.load_state_dict(torch.load("../experiments/m2v_music2.pth"))
    model.eval()

    wav = load_audio("Y:/separated/bt-music-1/100 Hits - Running Songs 2014 CD 2/100 Hits - Running Songs 2014 Cd2 - 02 - 7Th Heaven - Ain't Nothin' Goin' On But The Rent/00001/no_vocals.wav", 22050)
    mel = TorchMelSpectrogramInjector({'n_mel_channels': 256, 'mel_fmax': 11000, 'filter_length': 16000,
                                       'normalize': True, 'in': 'in', 'out': 'out'}, {})({'in': wav.unsqueeze(0)})['out']
    codes = model.get_codes(mel)
    reconstruction = model.reconstruct(mel)

    torchvision.utils.save_image((normalize_mel(mel).unsqueeze(1)+1)/2, 'mel.png')
    torchvision.utils.save_image((normalize_mel(reconstruction).unsqueeze(1)+1)/2, 'reconstructed.png')

    collapsed = collapse_codegroups(codes)
    recovered = recover_codegroups(collapsed, 4)

    print(codes)