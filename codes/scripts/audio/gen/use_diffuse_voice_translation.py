import argparse
import os

import torch
import torchaudio

from data.audio.unsupervised_audio_dataset import load_audio
from data.util import find_files_of_type, is_audio_file
from scripts.audio.gen.speech_synthesis_utils import do_spectrogram_diffusion, \
    load_discrete_vocoder_diffuser, wav_to_mel, convert_mel_to_codes
from utils.audio import plot_spectrogram
from utils.util import load_model_from_config, ceil_multiple
import torch.nn.functional as F


def get_ctc_codes_for(src_clip_path):
    """
    Uses wav2vec2 to infer CTC codes for the audio clip at the specified path.
    """
    from transformers import Wav2Vec2ForCTC
    from transformers import Wav2Vec2Processor
    model = Wav2Vec2ForCTC.from_pretrained(f"facebook/wav2vec2-large-960h").to("cuda")
    processor = Wav2Vec2Processor.from_pretrained(f"facebook/wav2vec2-large-960h")

    clip = load_audio(src_clip_path, 16000).squeeze()
    clip_inp = processor(clip.numpy(), return_tensors='pt', sampling_rate=16000).input_values.cuda()
    logits = model(clip_inp).logits
    return torch.argmax(logits, dim=-1), clip


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
    provided_voices = {
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
    parser.add_argument('-src_clip', type=str, help='Path to the audio files to translate', default='D:\\tortoise-tts\\voices')
    parser.add_argument('-opt', type=str, help='Path to options YAML file used to train the diffusion model', default='X:\\dlas\\experiments\\train_diffusion_tts5_medium\\train_diffusion_tts5_medium.yml')
    parser.add_argument('-diffusion_model_name', type=str, help='Name of the diffusion model in opt.', default='generator')
    parser.add_argument('-diffusion_model_path', type=str, help='Path to saved model weights', default='X:\\dlas\\experiments\\train_diffusion_tts5_medium\\models\\73000_generator_ema.pth')
    parser.add_argument('-sr_opt', type=str, help='Path to options YAML file used to train the SR diffusion model', default='X:\\dlas\\experiments\\train_diffusion_tts6_upsample.yml')
    parser.add_argument('-sr_diffusion_model_name', type=str, help='Name of the SR diffusion model in opt.', default='generator')
    parser.add_argument('-sr_diffusion_model_path', type=str, help='Path to saved model weights for the SR diffuser', default='X:\\dlas\\experiments\\train_diffusion_tts6_upsample_continued\\models\\53500_generator_ema.pth')
    parser.add_argument('-voice', type=str, help='Type of conditioning voice', default='plants')
    parser.add_argument('-diffusion_steps', type=int, help='Number of diffusion steps to perform to create the generate. Lower steps reduces quality, but >40 is generally pretty good.', default=100)
    parser.add_argument('-output_path', type=str, help='Where to store outputs.', default='../results/use_diffuse_voice_translation')
    parser.add_argument('-device', type=str, help='Device to run on', default='cuda')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # Fixed parameters.
    base_sample_rate = 5500
    sr_sample_rate = 22050

    print("Loading Diffusion Models..")
    diffusion = load_model_from_config(args.opt, args.diffusion_model_name, also_load_savepoint=False,
                                       load_path=args.diffusion_model_path, device='cpu').eval()
    diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=args.diffusion_steps, schedule='cosine')
    sr_diffusion = load_model_from_config(args.sr_opt, args.sr_diffusion_model_name, also_load_savepoint=False,
                                          load_path=args.sr_diffusion_model_path, device='cpu').eval()
    sr_diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=args.diffusion_steps, schedule='linear')
    sr_cond = load_audio(provided_voices[args.voice], sr_sample_rate).to(args.device)
    cond = torchaudio.functional.resample(sr_cond, sr_sample_rate, base_sample_rate)
    torchaudio.save(os.path.join(args.output_path, 'cond_base.wav'), cond.cpu(), base_sample_rate)
    torchaudio.save(os.path.join(args.output_path, 'cond_sr.wav'), sr_cond.cpu(), sr_sample_rate)

    with torch.no_grad():
        if os.path.isdir(args.src_clip):
            files = find_files_of_type('img', args.src_clip, qualifier=is_audio_file)[0]
        else:
            files = [args.src_clip]
        for e, file in enumerate(files):
            print("Extracting CTC codes from source clip..")
            aligned_codes, src_clip = get_ctc_codes_for(file)
            torchaudio.save(os.path.join(args.output_path, f'{e}_source_clip.wav'), src_clip.unsqueeze(0).cpu(), 16000)

            print("Performing initial diffusion..")
            output_shape, aligned_codes = determine_output_size(aligned_codes, base_sample_rate)
            diffusion = diffusion.cuda()
            output_base = diffuser.p_sample_loop(diffusion, output_shape, noise=torch.zeros(output_shape, device=args.device),
                                            model_kwargs={'tokens': aligned_codes,
                                            'conditioning_input': cond.unsqueeze(0)})
            diffusion = diffusion.cpu()
            torchaudio.save(os.path.join(args.output_path, f'{e}_output_mean_base.wav'), output_base.cpu().squeeze(0), base_sample_rate)

            print("Performing SR diffusion..")
            output_shape = (1, 1, output_base.shape[-1] * (sr_sample_rate // base_sample_rate))
            sr_diffusion = sr_diffusion.cuda()
            output = sr_diffuser.p_sample_loop(sr_diffusion, output_shape, #noise=torch.zeros(output_shape, device=args.device),
                                            model_kwargs={'tokens': aligned_codes,
                                            'conditioning_input': sr_cond.unsqueeze(0),
                                            'lr_input': output_base})
            sr_diffusion = sr_diffusion.cpu()
            torchaudio.save(os.path.join(args.output_path, f'{e}_output_mean_sr.wav'), output.cpu().squeeze(0), sr_sample_rate)
