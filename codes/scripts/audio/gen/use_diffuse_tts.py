import argparse
import os

import torch
import torchaudio

from data.audio.unsupervised_audio_dataset import load_audio
from scripts.audio.gen.speech_synthesis_utils import do_spectrogram_diffusion, \
    load_discrete_vocoder_diffuser, wav_to_mel, convert_mel_to_codes
from utils.audio import plot_spectrogram
from utils.util import load_model_from_config
import torch.nn.functional as F


def ceil_multiple(base, multiple):
    res = base % multiple
    if res == 0:
        return base
    return base + (multiple - res)


def determine_output_size(codes, base_sample_rate):
    aligned_codes_compression_factor = base_sample_rate * 221 // 11025
    output_size = codes.shape[-1]*aligned_codes_compression_factor
    padded_size = ceil_multiple(output_size, 2048)
    padding_added = padded_size - output_size
    padding_needed_for_codes = padding_added // aligned_codes_compression_factor
    if padding_needed_for_codes > 0:
        codes = F.pad(codes, (0, padding_needed_for_codes))
    output_shape = (1, 1, padded_size)
    return output_shape, codes


if __name__ == '__main__':
    conditioning_clips = {
        # Male
        'simmons': 'Y:\\clips\\books1\\754_Dan Simmons - The Rise Of Endymion 356 of 450\\00026.wav',
        'carlin': 'Y:\\clips\\books1\\12_dchha13 Bubonic Nukes\\00097.wav',
        'entangled': 'Y:\\clips\\books1\\3857_25_The_Entangled_Bank__000000000\\00123.wav',
        'snowden': 'Y:\\clips\\books1\\7658_Edward_Snowden_-_Permanent_Record__000000004\\00027.wav',
        'plants': 'Y:\\clips\\books1\\12028_The_Secret_Life_of_Plants_-_18__000000000\\00399.wav',
        # Female
        'the_doctor': 'Y:\\clips\\books2\\37062___The_Doctor__000000003\\00206.wav',
        'puppy': 'Y:\\clips\\books2\\17830___3_Puppy_Kisses__000000002\\00046.wav',
        'adrift': 'Y:\\clips\\books2\\5608_Gear__W_Michael_-_Donovan_1-5_(2018-2021)_(book_4_Gear__W_Michael_-_Donovan_5_-_Adrift_(2021)_Gear__W_Michael_-_Adrift_(Donovan_5)_—_82__000000000\\00019.wav',
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-text', type=str, help='Text to speak.', default='my father worked at the airport. he was air traffic control. he always knew when the president was flying in but was not allowed to tell anyone.')
    parser.add_argument('-opt_code_gen', type=str, help='Path to options YAML file used to train the code_gen model', default='D:\\dlas\\options\\train_encoder_build_ctc_alignments.yml')
    parser.add_argument('-code_gen_model_name', type=str, help='Name of the code_gen model in opt.', default='generator')
    parser.add_argument('-code_gen_model_path', type=str, help='Path to saved code_gen model weights', default='D:\\dlas\\experiments\\train_encoder_build_ctc_alignments_medium\\models\\50000_generator_ema.pth')
    parser.add_argument('-opt', type=str, help='Path to options YAML file used to train the diffusion model', default='X:\\dlas\\experiments\\train_diffusion_tts5_medium\\train_diffusion_tts5_medium.yml')
    parser.add_argument('-diffusion_model_name', type=str, help='Name of the diffusion model in opt.', default='generator')
    parser.add_argument('-diffusion_model_path', type=str, help='Path to saved model weights', default='X:\\dlas\\experiments\\train_diffusion_tts5_medium\\models\\73000_generator_ema.pth')
    parser.add_argument('-sr_opt', type=str, help='Path to options YAML file used to train the SR diffusion model', default='X:\\dlas\\experiments\\train_diffusion_tts6_upsample.yml')
    parser.add_argument('-sr_diffusion_model_name', type=str, help='Name of the SR diffusion model in opt.', default='generator')
    parser.add_argument('-sr_diffusion_model_path', type=str, help='Path to saved model weights for the SR diffuser', default='X:\\dlas\\experiments\\train_diffusion_tts6_upsample_continued\\models\\53500_generator_ema.pth')
    parser.add_argument('-cond', type=str, help='Type of conditioning voice', default='plants')
    parser.add_argument('-diffusion_steps', type=int, help='Number of diffusion steps to perform to create the generate. Lower steps reduces quality, but >40 is generally pretty good.', default=100)
    parser.add_argument('-output_path', type=str, help='Where to store outputs.', default='../results/use_diffuse_tts')
    parser.add_argument('-device', type=str, help='Device to run on', default='cuda')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # Fixed parameters.
    base_sample_rate = 5500
    sr_sample_rate = 22050

    print("Loading provided conditional audio..")
    sr_cond = load_audio(conditioning_clips[args.cond], sr_sample_rate).to(args.device)
    cond_mel = wav_to_mel(sr_cond)
    cond = torchaudio.functional.resample(sr_cond, sr_sample_rate, base_sample_rate)
    torchaudio.save(os.path.join(args.output_path, 'cond_base.wav'), cond.cpu(), base_sample_rate)
    torchaudio.save(os.path.join(args.output_path, 'cond_sr.wav'), sr_cond.cpu(), sr_sample_rate)

    print("Generating codes for text..")
    codegen = load_model_from_config(args.opt_code_gen, args.code_gen_model_name, also_load_savepoint=False,
                                       load_path=args.code_gen_model_path, device='cuda').eval()
    codes = codegen.generate(cond_mel, [args.text])
    del codegen

    print("Loading Diffusion Models..")
    diffusion = load_model_from_config(args.opt, args.diffusion_model_name, also_load_savepoint=False,
                                       load_path=args.diffusion_model_path, device='cpu').eval()
    diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=args.diffusion_steps, schedule='cosine')
    aligned_codes_compression_factor = base_sample_rate * 221 // 11025
    sr_diffusion = load_model_from_config(args.sr_opt, args.sr_diffusion_model_name, also_load_savepoint=False,
                                          load_path=args.sr_diffusion_model_path, device='cpu').eval()
    sr_diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=args.diffusion_steps, schedule='linear')

    with torch.no_grad():
        for p, code in enumerate([codes]):
            print("Loading data..")
            aligned_codes = code.to(args.device)

            print("Performing initial diffusion..")
            output_shape, aligned_codes = determine_output_size(aligned_codes, base_sample_rate)
            diffusion = diffusion.cuda()
            output_base = diffuser.p_sample_loop(diffusion, output_shape, noise=torch.zeros(output_shape, device=args.device),
                                            model_kwargs={'tokens': aligned_codes,
                                            'conditioning_input': cond.unsqueeze(0)})
            diffusion = diffusion.cpu()
            torchaudio.save(os.path.join(args.output_path, f'{p}_output_mean_base.wav'), output_base.cpu().squeeze(0), base_sample_rate)

            print("Performing SR diffusion..")
            output_shape = (1, 1, output_base.shape[-1] * (sr_sample_rate // base_sample_rate))
            sr_diffusion = sr_diffusion.cuda()
            output = sr_diffuser.p_sample_loop(sr_diffusion, output_shape, noise=torch.zeros(output_shape, device=args.device),
                                            model_kwargs={'tokens': torch.zeros_like(aligned_codes),
                                            'conditioning_input': sr_cond.unsqueeze(0),
                                            'lr_input': output_base})
            sr_diffusion = sr_diffusion.cpu()
            torchaudio.save(os.path.join(args.output_path, f'{p}_output_mean_sr.wav'), output.cpu().squeeze(0), sr_sample_rate)
