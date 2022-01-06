import argparse
import random

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from tokenizers import Tokenizer

from data.audio.paired_voice_audio_dataset import CharacterTokenizer
from data.audio.unsupervised_audio_dataset import load_audio
from data.util import is_audio_file, find_files_of_type
from models.tacotron2.text import text_to_sequence
from scripts.audio.gen.speech_synthesis_utils import do_spectrogram_diffusion, \
    load_discrete_vocoder_diffuser, wav_to_mel
from trainer.injectors.base_injectors import TorchMelSpectrogramInjector
from utils.options import Loader
from utils.util import load_model_from_config


# Loads multiple conditioning files at random from a folder.
def load_conditioning_candidates(path, num_conds, sample_rate=22050, cond_length=44100):
    candidates = find_files_of_type('img', path, qualifier=is_audio_file)[0]
    # Sample with replacement. This can get repeats, but more conveniently handles situations where there are not enough candidates.
    related_mels = []
    for k in range(num_conds):
        rel_clip = load_audio(candidates[k], sample_rate)
        gap = rel_clip.shape[-1] - cond_length
        if gap < 0:
            rel_clip = F.pad(rel_clip, pad=(0, abs(gap)))
        elif gap > 0:
            rand_start = random.randint(0, gap)
            rel_clip = rel_clip[:, rand_start:rand_start + cond_length]
        mel_clip = wav_to_mel(rel_clip.unsqueeze(0)).squeeze(0)
        related_mels.append(mel_clip)
    return torch.stack(related_mels, dim=0).unsqueeze(0).cuda(), rel_clip.unsqueeze(0).cuda()


def load_conditioning(path, sample_rate=22050, cond_length=44100):
    rel_clip = load_audio(path, sample_rate)
    gap = rel_clip.shape[-1] - cond_length
    if gap < 0:
        rel_clip = F.pad(rel_clip, pad=(0, abs(gap)))
    elif gap > 0:
        rand_start = random.randint(0, gap)
        rel_clip = rel_clip[:, rand_start:rand_start + cond_length]
    mel_clip = wav_to_mel(rel_clip.unsqueeze(0)).squeeze(0)
    return mel_clip.unsqueeze(0).cuda(), rel_clip.unsqueeze(0).cuda()


def fix_autoregressive_output(codes, stop_token):
    """
    This function performs some padding on coded audio that fixes a mismatch issue between what the diffusion model was
    trained on and what the autoregressive code generator creates (which has no padding or end).
    This is highly specific to the DVAE being used, so this particular coding will not necessarily work if used with
    a different DVAE. This can be inferred by feeding a audio clip padded with lots of zeros on the end through the DVAE
    and copying out the last few codes.

    Failing to do this padding will produce speech with a harsh end that sounds like "BLAH" or similar.
    """
    # Strip off the autoregressive stop token and add padding.
    stop_token_indices = (codes == stop_token).nonzero()
    if len(stop_token_indices) == 0:
        print("No stop tokens found, enjoy that output of yours!")
    else:
        codes = codes[:stop_token_indices[0]]

    padding = torch.tensor([83,   83,   83,  83,   83,   83,  83,   83,   83,   45,   45,  248],
                           dtype=torch.long, device=codes.device)
    return torch.cat([codes, padding])


if __name__ == '__main__':
    preselected_cond_voices = {
        'trump': 'D:\\data\\audio\\sample_voices\\trump.wav',
        'ryan_reynolds': 'D:\\data\\audio\\sample_voices\\ryan_reynolds.wav',
        'ed_sheeran': 'D:\\data\\audio\\sample_voices\\ed_sheeran.wav',
        'simmons': 'Y:\\clips\\books1\\754_Dan Simmons - The Rise Of Endymion 356 of 450\\00026.wav',
        'news_girl': 'Y:\\clips\\podcasts-0\\8288_20210113-Is More Violence Coming_\\00022.wav',
        'dan_carlin': 'Y:\\clips\\books1\5_dchha06 Shield of the West\\00476.wav',
        'libri_test': 'Z:\\bigasr_dataset\\libritts\\test-clean\\672\\122797\\672_122797_000057_000002.wav'
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt_diffuse', type=str, help='Path to options YAML file used to train the diffusion model', default='X:\\dlas\\experiments\\train_diffusion_vocoder_with_cond_new_dvae.yml')
    parser.add_argument('-diffusion_model_name', type=str, help='Name of the diffusion model in opt.', default='generator')
    parser.add_argument('-diffusion_model_path', type=str, help='Diffusion model checkpoint to load.', default='X:\\dlas\\experiments\\train_diffusion_vocoder_with_cond_new_dvae_full\\models\\6100_generator_ema.pth')
    parser.add_argument('-dvae_model_name', type=str, help='Name of the DVAE model in opt.', default='dvae')
    parser.add_argument('-opt_gpt_tts', type=str, help='Path to options YAML file used to train the GPT-TTS model', default='X:\\dlas\\experiments\\train_gpt_unified_finetune_tts.yml')
    parser.add_argument('-gpt_tts_model_name', type=str, help='Name of the GPT TTS model in opt.', default='gpt')
    parser.add_argument('-gpt_tts_model_path', type=str, help='GPT TTS model checkpoint to load.', default='X:\\dlas\\experiments\\train_gpt_unified_finetune_tts_libri_all_and_hifi_no_unsupervised\\models\\17500_gpt.pth')
    parser.add_argument('-text', type=str, help='Text to speak.', default="I am a language model that has learned to speak.")
    parser.add_argument('-cond_path', type=str, help='Path to condioning sample.', default='')
    parser.add_argument('-cond_preset', type=str, help='Use a preset conditioning voice (defined above). Overrides cond_path.', default='libri_test')
    parser.add_argument('-num_samples', type=int, help='How many outputs to produce.', default=1)
    args = parser.parse_args()
    # libritts_text = 'fall passed so quickly, there was so much going on around him, the tree quite forgot to look to himself.'

    print("Loading GPT TTS..")
    with open(args.opt_gpt_tts, mode='r') as f:
        gpt_opt = yaml.load(f, Loader=Loader)
    gpt_opt['networks'][args.gpt_tts_model_name]['kwargs']['checkpointing'] = False  # Required for beam search
    gpt = load_model_from_config(preloaded_options=gpt_opt, model_name=args.gpt_tts_model_name, also_load_savepoint=False, load_path=args.gpt_tts_model_path, strict_load=False)

    print("Loading data..")
    tokenizer = CharacterTokenizer()
    text = torch.IntTensor(tokenizer.encode(args.text)).unsqueeze(0).cuda()
    text = F.pad(text, (0,1))  # This may not be necessary.
    paired_text_length = gpt_opt['datasets']['train']['max_paired_text_length']
    assert paired_text_length >= text.shape[1]

    cond_path = args.cond_path if args.cond_preset is None else preselected_cond_voices[args.cond_preset]
    conds, cond_wav = load_conditioning(cond_path)

    print("Performing GPT inference..")
    codes = gpt.inference_speech(conds, text, num_beams=1, repetition_penalty=1.0, do_sample=True, top_k=20, top_p=.95,
                          num_return_sequences=args.num_samples, length_penalty=1, early_stopping=True)

    # Delete the GPT TTS model to free up GPU memory
    stop_token = gpt.stop_mel_token
    del gpt

    print("Loading DVAE..")
    dvae = load_model_from_config(args.opt_diffuse, args.dvae_model_name)
    print("Loading Diffusion Model..")
    diffusion = load_model_from_config(args.opt_diffuse, args.diffusion_model_name, also_load_savepoint=False, load_path=args.diffusion_model_path)
    diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=50)

    print("Performing vocoding..")
    # Perform vocoding on each batch element separately: Vocoding is very memory intensive.
    for b in range(codes.shape[0]):
        code = fix_autoregressive_output(codes[b], stop_token).unsqueeze(0)
        wav = do_spectrogram_diffusion(diffusion, dvae, diffuser, code, cond_wav,
                                       spectrogram_compression_factor=128, plt_spec=False)
        torchaudio.save(f'gpt_tts_output_{b}.wav', wav.squeeze(0).cpu(), 11025)
