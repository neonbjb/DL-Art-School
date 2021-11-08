import os

import numpy
import torch
import torch.nn as nn
from matplotlib import pyplot
from torch.utils.tensorboard import SummaryWriter

from data.audio.unsupervised_audio_dataset import load_audio
from models.gpt_voice.gpt_asr_hf import GptAsrHf
from models.tacotron2.text import text_to_sequence
from trainer.injectors.base_injectors import MelSpectrogramInjector

if __name__ == '__main__':
    audio_data = load_audio('Z:\\split\\classified\\fine\\books1\\2_dchha03 The Organization of Peace\\00010.wav', 22050).unsqueeze(0)
    audio_data = torch.nn.functional.pad(audio_data, (0, 358395-audio_data.shape[-1]))
    mel_inj = MelSpectrogramInjector({'in': 'in', 'out': 'out'}, {})
    mel = mel_inj({'in': audio_data})['out'].cuda()
    actual_text = 'and it doesn\'t take very long.'
    labels = torch.IntTensor(text_to_sequence(actual_text, ['english_cleaners'])).unsqueeze(0).cuda()

    model = GptAsrHf(layers=12, model_dim=512, max_mel_frames=1400, max_symbols_per_phrase=250, heads=8)
    model.load_state_dict(torch.load('X:\\dlas\\experiments\\train_gpt_asr_mass_hf\\models\\31000_gpt_ema.pth'))
    model = model.cuda()

    with torch.no_grad():
        attentions = model(mel, labels, return_attentions=True)
        attentions = torch.stack(attentions, dim=0).permute(0,1,2,4,3)[:, :, :, -model.max_symbols_per_phrase:, :model.max_mel_frames]
        attentions = attentions.sum(0).sum(1).squeeze()

    xs = [str(i) for i in range(1, model.max_mel_frames+1, 1)]
    os.makedirs('results', exist_ok=True)
    logger = SummaryWriter('results')
    for e, character_attn in enumerate(attentions):
        if e >= len(actual_text):
            break
        fig = pyplot.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(xs, character_attn.cpu().numpy())
        logger.add_figure(f'{e}_{actual_text[e]}', fig)
