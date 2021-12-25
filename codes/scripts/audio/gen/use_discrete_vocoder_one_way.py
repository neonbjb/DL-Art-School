import argparse

import torch
import torchaudio

from data.audio.unsupervised_audio_dataset import load_audio
from scripts.audio.gen.speech_synthesis_utils import do_spectrogram_diffusion, \
    load_discrete_vocoder_diffuser, wav_to_mel, convert_mel_to_codes
from utils.audio import plot_spectrogram
from utils.util import load_model_from_config


def roundtrip_vocoding(dvae, vocoder, diffuser, clip, cond=None, plot_spec=False):
    clip = clip.unsqueeze(0)
    if cond is None:
        cond = clip
    else:
        cond = cond.unsqueeze(0)
    mel = wav_to_mel(clip)
    if plot_spec:
        plot_spectrogram(mel[0].cpu())
    codes = convert_mel_to_codes(dvae, mel)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file used to train the diffusion model', default='X:\\dlas\\experiments\\train_diffusion_vocoder_with_cond_new_dvae.yml')
    parser.add_argument('-diffusion_model_name', type=str, help='Name of the diffusion model in opt.', default='generator')
    parser.add_argument('-diffusion_model_path', type=str, help='Name of the diffusion model in opt.', default='X:\\dlas\\experiments\\train_diffusion_vocoder_with_cond_new_dvae_full\\models\\6100_generator_ema.pth')
    parser.add_argument('-dvae_model_name', type=str, help='Name of the DVAE model in opt.', default='dvae')
    parser.add_argument('-input_file', type=str, help='Path to the input torch save file.', default='speech_forward_mels.pth')
    parser.add_argument('-cond', type=str, help='Path to the conditioning input audio file.', default='Z:\\clips\\books1\\3042_18_Holden__000000000\\00037.wav')
    args = parser.parse_args()

    print("Loading DVAE..")
    dvae = load_model_from_config(args.opt, args.dvae_model_name)
    print("Loading Diffusion Model..")
    diffusion = load_model_from_config(args.opt, args.diffusion_model_name, also_load_savepoint=False, load_path=args.diffusion_model_path)

    print("Loading data..")
    cond = load_audio(args.cond, 22050)
    if cond.shape[-1] > 44100+10000:
        cond = cond[:,10000:54100]
    cond = cond.unsqueeze(0).cuda()

    diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=20)
    inp = torch.load(args.input_file)
    codes = inp

    print("Performing inference..")
    for i, cb in enumerate(codes):
        roundtripped = do_spectrogram_diffusion(diffusion, dvae, diffuser, cb.unsqueeze(0).cuda(), cond, spectrogram_compression_factor=128, plt_spec=False)
        torchaudio.save(f'vocoded_output_sp_{i}.wav', roundtripped.squeeze(0).cpu(), 11025)