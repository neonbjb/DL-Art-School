import argparse

import torch
import torchaudio

from data.audio.unsupervised_audio_dataset import load_audio
from scripts.audio.gen.speech_synthesis_utils import do_spectrogram_diffusion, \
    load_discrete_vocoder_diffuser, wav_to_mel, convert_mel_to_codes, wav_to_univnet_mel, load_univnet_vocoder
from trainer.injectors.audio_injectors import denormalize_mel
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
    return do_spectrogram_diffusion(vocoder, dvae, diffuser, codes, cond, spectrogram_compression_factor=256, plt_spec=plot_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-codes_file', type=str, help='Which discretes to decode. Should be a path to a pytorch pickle that simply contains the codes.')
    parser.add_argument('-cond_file', type=str, help='Path to the input audio file.')
    parser.add_argument('-opt', type=str, help='Path to options YAML file used to train the diffusion model',
                        default='X:\\dlas\\experiments\\train_diffusion_tts_mel_flat0\\last_train.yml')
    parser.add_argument('-diffusion_model_name', type=str, help='Name of the diffusion model in opt.', default='generator')
    parser.add_argument('-diffusion_model_path', type=str, help='Diffusion model checkpoint to load.', default='X:\\dlas\\experiments\\train_diffusion_tts_mel_flat0\\models\\114000_generator_ema.pth')
    args = parser.parse_args()

    print("Loading data..")
    codes = torch.load(args.codes_file)
    conds = load_audio(args.cond_file, 24000)
    conds = conds[:,:102400]
    cond_mel = wav_to_univnet_mel(conds.to('cuda'), do_normalization=False)
    output_shape = (1,100,codes.shape[-1]*4)

    print("Loading Diffusion Model..")
    diffusion = load_model_from_config(args.opt, args.diffusion_model_name, also_load_savepoint=False, load_path=args.diffusion_model_path, strict_load=False).cuda().eval()
    diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=50, schedule='linear', enable_conditioning_free_guidance=True, conditioning_free_k=1)
    vocoder = load_univnet_vocoder().cuda()

    with torch.no_grad():
        print("Performing inference..")
        for i in range(codes.shape[0]):
            gen_mel = diffuser.p_sample_loop(diffusion, output_shape, model_kwargs={'aligned_conditioning': codes[i].unsqueeze(0), 'conditioning_input': cond_mel})
            gen_mel = denormalize_mel(gen_mel)
            genWav = vocoder.inference(gen_mel)
            torchaudio.save(f'vocoded_{i}.wav', genWav.cpu().squeeze(0), 24000)