import argparse
import os

import torch
import torchaudio

from data.audio.unsupervised_audio_dataset import load_audio
from scripts.audio.gen.speech_synthesis_utils import do_spectrogram_diffusion, \
    load_discrete_vocoder_diffuser, wav_to_mel, convert_mel_to_codes
from utils.audio import plot_spectrogram
from utils.util import load_model_from_config


def ceil_multiple(base, multiple):
    res = base % multiple
    if res == 0:
        return base
    return base + (multiple - res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file used to train the diffusion model', default='../options/train_diffusion_tts_medium.yml')
    parser.add_argument('-diffusion_model_name', type=str, help='Name of the diffusion model in opt.', default='generator')
    parser.add_argument('-diffusion_model_path', type=str, help='Path to saved model weights', default='X:\\dlas\\experiments\\train_diffusion_tts_medium\\models\\14800_generator_ema.pth')
    parser.add_argument('-aligned_codes', type=str, help='Comma-delimited list of integer codes that defines text & prosody. Get this by apply W2V to an existing audio clip or from a bespoke generator.',
                        default='0,0,0,0,10,10,0,4,0,7,0,17,4,4,0,25,5,0,13,13,0,22,4,4,0,21,15,15,7,0,0,14,4,4,6,8,4,4,0,0,12,5,0,0,5,0,4,4,22,22,8,16,16,0,4,4,4,0,0,0,0,0,0,0')  # Default: 'i am very glad to see you', libritts/train-clean-100/103/1241/103_1241_000017_000001.wav.
    # -cond "Y:\libritts/train-clean-100/103/1241/103_1241_000017_000001.wav"
    parser.add_argument('-cond', type=str, help='Path to the conditioning input audio file.', default='Y:\\clips\\books1\\754_Dan Simmons - The Rise Of Endymion 356 of 450\\00026.wav')
    parser.add_argument('-diffusion_steps', type=int, help='Number of diffusion steps to perform to create the generate. Lower steps reduces quality, but >40 is generally pretty good.', default=100)
    parser.add_argument('-output_path', type=str, help='Where to store outputs.', default='../results/use_diffuse_tts')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    print("Loading Diffusion Model..")
    diffusion = load_model_from_config(args.opt, args.diffusion_model_name, also_load_savepoint=False, load_path=args.diffusion_model_path)
    aligned_codes_compression_factor = 221  # Derived empirically for 11025Hz sample rate. Adjust if sample rate increases.

    print("Loading data..")
    aligned_codes = torch.tensor([int(s) for s in args.aligned_codes.split(',')]).cuda()
    diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=args.diffusion_steps)
    cond = load_audio(args.cond, 22050).cuda()
    if cond.shape[-1] > 88000:
        cond = cond[:,:88000]

    with torch.no_grad():
        print("Performing inference..")
        diffusion.eval()
        output_shape = (1, 1, ceil_multiple(aligned_codes.shape[-1]*aligned_codes_compression_factor, 2048))

        output = diffuser.p_sample_loop(diffusion, output_shape, noise=torch.zeros(output_shape, device='cuda'),
                                        model_kwargs={'tokens': aligned_codes.unsqueeze(0),
                                        'conditioning_input': cond.unsqueeze(0)})
        torchaudio.save(os.path.join(args.output_path, f'output_mean.wav'), output.cpu().squeeze(0), 11025)

        for k in range(5):
            output = diffuser.p_sample_loop(diffusion, output_shape, model_kwargs={'tokens': aligned_codes.unsqueeze(0),
                                                                                   'conditioning_input': cond.unsqueeze(0)})

            torchaudio.save(os.path.join(args.output_path, f'output_{k}.wav'), output.cpu().squeeze(0), 11025)
