import argparse
import os
import random

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from tqdm import tqdm

from data.audio.unsupervised_audio_dataset import load_audio
from data.audio.voice_tokenizer import VoiceBpeTokenizer
from data.util import is_audio_file, find_files_of_type
from scripts.audio.gen.speech_synthesis_utils import do_spectrogram_diffusion, \
    load_discrete_vocoder_diffuser, wav_to_mel
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
        return
    else:
        codes[stop_token_indices] = 83
    stm = stop_token_indices.min().item()
    codes[stm:] = 83
    if stm - 3 < codes.shape[0]:
        codes[-3] = 45
        codes[-2] = 45
        codes[-1] = 248

    return codes


if __name__ == '__main__':
    preselected_cond_voices = {
        'trump': ['D:\\data\\audio\\sample_voices\\trump.wav'],
        'obama': ['D:\\data\\audio\\sample_voices\\obama1.mp3', 'D:\\data\\audio\\sample_voices\\obama2.wav'],
        'ryan_reynolds': ['D:\\data\\audio\\sample_voices\\ryan_reynolds.wav'],
        'ed_sheeran': ['D:\\data\\audio\\sample_voices\\ed_sheeran.wav'],
        'simmons': ['Y:\\clips\\books1\\754_Dan Simmons - The Rise Of Endymion 356 of 450\\00026.wav'],
        'news_girl': ['Y:\\clips\\podcasts-0\\8288_20210113-Is More Violence Coming_\\00022.wav', 'Y:\\clips\\podcasts-0\\8288_20210113-Is More Violence Coming_\\00016.wav'],
        'dan_carlin': ['Y:\\clips\\books1\\5_dchha06 Shield of the West\\00476.wav', 'Y:\\clips\\books1\\15_dchha16 Nazi Tidbits\\00036.wav'],
        'libri_test': ['Y:\\libritts\\test-clean\\672\\122797\\672_122797_000057_000002.wav'],
        'myself': ['D:\\data\\audio\\sample_voices\\myself1.wav', 'D:\\data\\audio\\sample_voices\\myself2.wav'],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt_diffuse', type=str, help='Path to options YAML file used to train the diffusion model', default='X:\\dlas\\experiments\\train_diffusion_vocoder_22k_level.yml')
    parser.add_argument('-diffusion_model_name', type=str, help='Name of the diffusion model in opt.', default='generator')
    parser.add_argument('-diffusion_model_path', type=str, help='Diffusion model checkpoint to load.', default='X:\\dlas\\experiments\\train_diffusion_vocoder_22k_level\\models\\15000_generator_ema.pth')
    parser.add_argument('-dvae_model_name', type=str, help='Name of the DVAE model in opt.', default='dvae')
    parser.add_argument('-opt_gpt_tts', type=str, help='Path to options YAML file used to train the GPT-TTS model', default='X:\\dlas\\experiments\\train_gpt_tts_unified.yml')
    parser.add_argument('-gpt_tts_model_name', type=str, help='Name of the GPT TTS model in opt.', default='gpt')
    parser.add_argument('-gpt_tts_model_path', type=str, help='GPT TTS model checkpoint to load.', default='X:\\dlas\\experiments\\train_gpt_tts_unified_large\\models\\45000_gpt_ema.pth')
    parser.add_argument('-opt_clip', type=str, help='Path to options YAML file used to train the CLIP model', default='X:\\dlas\\experiments\\train_clip_text_to_voice.yml')
    parser.add_argument('-clip_model_name', type=str, help='Name of the CLIP model in opt.', default='clip')
    parser.add_argument('-clip_model_path', type=str, help='CLIP model checkpoint to load.', default='X:\\dlas\\experiments\\train_clip_text_to_voice_masking_bigger_batch\\models\\23500_clip_ema.pth')
    parser.add_argument('-opt_cond_clip', type=str, help='Path to options YAML file used to train the Conditioning CLIP model', default='D:\\dlas\\options\\train_clip_cond_to_voice.yml')
    parser.add_argument('-cond_clip_model_name', type=str, help='Name of the CLIP model in opt.', default='clip')
    parser.add_argument('-cond_clip_model_path', type=str, help='CLIP model checkpoint to load.', default='D:\\dlas\\experiments\\train_clip_cond_to_voice\\models\\42000_clip_ema.pth')
    parser.add_argument('-cond_clip_weight', type=float, help='How much to weight the conditioning CLIP to the text CLIP. Lower means the sample sounds more like the text, higher means it sounds more like the conditioning.',
                        default=.3)
    parser.add_argument('-text', type=str, help='Text to speak.', default="I am a language model that has learned to speak.")
    parser.add_argument('-cond_preset', type=str, help='Use a preset conditioning voice (defined above). Overrides cond_path.', default='simmons')
    parser.add_argument('-num_samples', type=int, help='How many total outputs the autoregressive transformer should produce.', default=256)
    parser.add_argument('-num_batches', type=int, help='How many batches those samples should be produced over.', default=16)
    parser.add_argument('-num_outputs', type=int, help='Number of outputs to produce.', default=5)
    parser.add_argument('-output_path', type=str, help='Where to store outputs.', default='../results/use_gpt_tts')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    # libritts_text = 'fall passed so quickly, there was so much going on around him, the tree quite forgot to look to himself.'

    print("Loading GPT TTS..")
    with open(args.opt_gpt_tts, mode='r') as f:
        gpt_opt = yaml.load(f, Loader=Loader)
    gpt_opt['networks'][args.gpt_tts_model_name]['kwargs']['checkpointing'] = False  # Required for beam search
    gpt = load_model_from_config(preloaded_options=gpt_opt, model_name=args.gpt_tts_model_name, also_load_savepoint=False, load_path=args.gpt_tts_model_path).cuda().eval()
    stop_mel_token = gpt.stop_mel_token

    print("Loading data..")
    tokenizer = VoiceBpeTokenizer('../experiments/bpe_lowercase_asr_256.json')
    text = torch.IntTensor(tokenizer.encode(args.text)).unsqueeze(0).cuda()
    text = F.pad(text, (0,1))  # This may not be necessary.

    cond_paths = preselected_cond_voices[args.cond_preset]
    conds = []
    for cond_path in cond_paths:
        c, cond_wav = load_conditioning(cond_path, cond_length=132300)
        conds.append(c)
    conds = torch.stack(conds, dim=1)  # And just use the last cond_wav for the diffusion model.

    with torch.no_grad():
        print("Performing GPT inference..")
        samples = []
        ctc_codes = []
        samples_per_batch = args.num_samples//args.num_batches
        for b in tqdm(range(args.num_batches)):
            codes, attentions = gpt.inference_speech(conds, text, num_beams=1, repetition_penalty=1.0, do_sample=True, top_k=50, top_p=.95,
                                                     temperature=.9, num_return_sequences=samples_per_batch, length_penalty=1,
                                                     return_attentions=True)
            padding_needed = 250 - codes.shape[1]
            codes = F.pad(codes, (0, padding_needed), value=stop_mel_token)
            samples.append(codes)
            ctc_codes.extend(gpt.convert_attentions_to_aligned_codes(text, attentions, codes, conds.shape[1]))
        samples = torch.cat(samples, dim=0)

        print("Loading CLIP..")
        clip = load_model_from_config(args.opt_clip, model_name=args.clip_model_name, also_load_savepoint=False, load_path=args.clip_model_path).cuda().eval()
        cond_clip = load_model_from_config(args.opt_cond_clip, model_name=args.cond_clip_model_name, also_load_savepoint=False, load_path=args.cond_clip_model_path).cuda().eval()
        print("Performing CLIP filtering..")
        for i in range(samples.shape[0]):
            samples[i] = fix_autoregressive_output(samples[i], stop_mel_token)
        clip_results = clip(text.repeat(samples.shape[0], 1),
                            torch.full((samples.shape[0],), fill_value=text.shape[1]-1, dtype=torch.long, device='cuda'),
                            samples, torch.full((samples.shape[0],), fill_value=samples.shape[1]*1024, dtype=torch.long, device='cuda'),
                            return_loss=False)
        cond_clip_results = cond_clip(conds[:, -1], samples, torch.full((samples.shape[0],), fill_value=samples.shape[1]*1024,
                                                                        dtype=torch.long, device='cuda'), return_loss=False)
        clip_results = clip_results * (1-args.cond_clip_weight) + cond_clip_results * args.cond_clip_weight
        best_indices = torch.topk(clip_results, k=args.num_outputs).indices
        best_results = samples[best_indices]
        best_codes = [ctc_codes[i] for i in best_indices]

        # Delete the GPT TTS and associated models to free up GPU memory before diffusion.
        del samples, clip, gpt

        print("Loading DVAE..")
        dvae = load_model_from_config(args.opt_diffuse, args.dvae_model_name).cuda()
        print("Loading Diffusion Model..")
        diffusion = load_model_from_config(args.opt_diffuse, args.diffusion_model_name, also_load_savepoint=False, load_path=args.diffusion_model_path).cuda()
        diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=100)

        print("Performing vocoding..")
        # Perform vocoding on each batch element separately: Vocoding is very memory intensive.
        for b in range(best_results.shape[0]):
            code = best_results[b].unsqueeze(0)
            wav = do_spectrogram_diffusion(diffusion, dvae, diffuser, code, cond_wav,
                                           spectrogram_compression_factor=256, plt_spec=False)
            torchaudio.save(os.path.join(args.output_path, f'gpt_tts_output_{b}.wav'), wav.squeeze(0).cpu(), 22050)
