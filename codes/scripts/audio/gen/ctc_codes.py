from itertools import groupby

import torch
import torchaudio
from transformers import Wav2Vec2CTCTokenizer

from data.audio.voice_tokenizer import VoiceBpeTokenizer
from models.audio.tts.ctc_code_generator import CtcCodeGenerator
from models.audio.tts.transformer_diffusion_tts import TransformerDiffusionTTS
from scripts.audio.gen.speech_synthesis_utils import load_discrete_vocoder_diffuser, load_univnet_vocoder, load_clvp
from trainer.injectors.audio_injectors import TorchMelSpectrogramInjector, denormalize_mel
from utils.util import load_audio


def get_ctc_metadata(codes):
    if isinstance(codes, torch.Tensor):
        codes = codes.tolist()
    grouped = groupby(codes)
    rcodes, repeats, pads = [], [], [0]
    for val, group in grouped:
        if val == 0:
            pads[-1] = len(list(
                group))  # This is a very important distinction! It means the padding belongs to the character proceeding it.
        else:
            rcodes.append(val)
            repeats.append(len(list(group)))
            pads.append(0)

    rcodes = torch.tensor(rcodes)
    # These clip values are sane maximum values which I did not see in the datasets I have access to.
    repeats = torch.clip(torch.tensor(repeats), min=1, max=30)
    pads = torch.clip(torch.tensor(pads[:-1]), max=120)

    return rcodes, pads, repeats


def decode_ctc_metadata(rcodes, pads, repeats):
    outp = []
    for s in range(rcodes.shape[-1]):
        outp = outp + [0 for _ in range(pads[s])]
        outp = outp + [rcodes[s].item() for _ in range(repeats[s])]
    return torch.tensor(outp, device=rcodes.device)


def diffuse(text, codes, cond):
    RATIO = 263/140

    codes = codes.cuda();
    cond = cond.cuda()

    bpe_tokenizer = VoiceBpeTokenizer('../experiments/bpe_lowercase_asr_256.json')
    clvp = load_clvp().cuda()
    diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=200, schedule='linear',
                                             enable_conditioning_free_guidance=False,
                                             conditioning_free_k=1)
    diffusion_model = TransformerDiffusionTTS(model_channels=896, num_layers=16, in_channels=100, in_latent_channels=1024,
                                              token_count=256, out_channels=200, dropout=0, unconditioned_percentage=0)
    diffusion_model.load_state_dict(torch.load('X:\\dlas\\experiments\\train_speech_diffusion_from_ctc_tfd5\\models\\26500_generator_ema.pth'))
    diffusion_model = diffusion_model.cuda().eval()
    with torch.no_grad():
        text_codes = torch.LongTensor(bpe_tokenizer.encode(text)).unsqueeze(0).to(codes.device)
        clvp_latent = clvp.embed_text(text_codes)
        cond_mel = TorchMelSpectrogramInjector({'n_mel_channels': 100, 'mel_fmax': 11000, 'filter_length': 8000, 'normalize': True,
                                                    'true_normalization': True, 'in': 'in', 'out': 'out'}, {})({'in': cond})['out']
        gen = diffuser.p_sample_loop(diffusion_model, (1,100,int(codes.shape[-1]*RATIO)), model_kwargs={'codes': codes,
                                                                                                   'conditioning_input': cond_mel,
                                                                                                   'type': torch.tensor([0], device=codes.device),
                                                                                                   'clvp_input': clvp_latent})
        gen_denorm = denormalize_mel(gen)
        vocoder = load_univnet_vocoder().cuda()
        gen_wav = vocoder.inference(gen_denorm)
    return gen_wav


if __name__ == '__main__':
    model = CtcCodeGenerator(model_dim=512, layers=16, dropout=0).eval().cuda()
    model.load_state_dict(torch.load('../experiments/train_encoder_build_ctc_alignments_toy/models/76000_generator_ema.pth'))

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('jbetker/tacotron-symbols')
    text = "Can I have tea and a pot of butter, please?"
    #seq = [0, 0, 0, 38, 51, 51, 41, 11, 11, 51, 51, 0, 0, 0, 0, 52, 0, 60, 0, 0, 0, 0, 0, 0, 6, 11, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 45, 0, 38, 57, 57, 11, 0, 41, 52, 52, 11, 11, 62, 52, 52, 58, 0, 11, 11, 60, 0, 0, 0, 0, 38, 0, 0, 51, 51, 0, 0, 57, 0, 0, 7, 7, 0, 0, 0]
    #codes, pads, repeats = get_ctc_metadata(seq)
    codes = tokenizer.encode(text)

    with torch.no_grad():
        codes = torch.tensor(codes).cuda().unsqueeze(0)
        ppads = torch.zeros_like(codes)
        prepeats = torch.zeros_like(codes)
        mask = torch.zeros_like(codes)
        for s in range(codes.shape[-1]):
            logits, confidences = model.inference(codes, ppads * mask, prepeats * mask)

            confidences = confidences * mask.logical_not()  # prevent prediction of tokens that have already been predicted.
            i = confidences.argmax(dim=-1)
            pred = logits[0,i].argmax()

            pred_pads = pred % model.max_pad
            pred_repeats = pred // model.max_pad
            ppads[0,i] = pred_pads
            prepeats[0,i] = pred_repeats
            mask[0,i] = 1

            #print(f"conf: {conf_str} pads={pred_pads}:{pads[0,i].item()} repeats={pred_repeats}:{repeats[0,i].item()}")

        decoded_codes = decode_ctc_metadata(codes[0], ppads[0], prepeats[0]).unsqueeze(0)
        cond = load_audio('D:\\tortoise-tts\\tortoise\\voices\\train_dotrice\\1.wav', 22050).unsqueeze(0).cuda()
        decoded_wav = diffuse(text, decoded_codes, cond)
        torchaudio.save('output.wav', decoded_wav.cpu()[0], 24000)