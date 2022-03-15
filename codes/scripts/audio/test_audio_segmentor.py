import os.path as osp
import logging
import random
import argparse

import audio2numpy
from munch import munchify

import utils
import utils.options as option
import utils.util as util
from data.audio.nv_tacotron_dataset import save_mel_buffer_to_file
from models.audio.tts.tacotron2 import hparams
from models.audio.tts.tacotron2 import TacotronSTFT
from models.audio.tts.tacotron2 import sequence_to_text
from scripts.audio.use_vocoder import Vocoder
from trainer.ExtensibleTrainer import ExtensibleTrainer
import torch
import numpy as np
from scipy.io import wavfile


def forward_pass(model, data, output_dir, opt, b):
    with torch.no_grad():
        model.feed_data(data, 0)
        model.test()

    if 'real_text' in opt['eval'].keys():
        real = data[opt['eval']['real_text']][0]
        print(f'{b} Real text: "{real}"')

    pred_seq = model.eval_state[opt['eval']['gen_text']][0]
    pred_text = [sequence_to_text(ts) for ts in pred_seq]
    audio = model.eval_state[opt['eval']['audio']][0].cpu().numpy()
    wavfile.write(osp.join(output_dir, f'{b}_clip.wav'), 22050, audio)
    for i, text in enumerate(pred_text):
        print(f'{b} Predicted text {i}: "{text}"')


if __name__ == "__main__":
    input_file = "E:\\audio\\books\\Roald Dahl Audiobooks\\Roald Dahl - The BFG\\(Roald Dahl) The BFG - 07.mp3"
    config = "../options/train_gpt_stop_libritts.yml"
    cutoff_pred_percent = .2

    # Set seeds
    torch.manual_seed(5555)
    random.seed(5555)
    np.random.seed(5555)

    #### options
    torch.backends.cudnn.benchmark = True
    want_metrics = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default=config)
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    utils.util.loaded_options = opt
    hp = munchify(hparams.create_hparams())

    util.mkdirs(
        (path for key, path in opt['path'].items()
         if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    model = ExtensibleTrainer(opt)
    assert len(model.networks) == 1
    model = model.networks[next(iter(model.networks.keys()))].module.to('cuda')
    model.eval()

    vocoder = Vocoder()

    audio, sr = audio2numpy.audio_from_file(input_file)
    if len(audio.shape) == 2:
        audio = audio[:, 0]
    audio = torch.tensor(audio, device='cuda').unsqueeze(0).unsqueeze(0)
    audio = torch.nn.functional.interpolate(audio, scale_factor=hp.sampling_rate/sr, mode='nearest').squeeze(1)
    stft = TacotronSTFT(hp.filter_length, hp.hop_length, hp.win_length, hp.n_mel_channels, hp.sampling_rate, hp.mel_fmin, hp.mel_fmax).to('cuda')
    mels = stft.mel_spectrogram(audio)

    with torch.no_grad():
        sentence_number = 0
        last_detection_start = 0
        start = 0
        clip_size = model.max_mel_frames
        while start+clip_size < mels.shape[-1]:
            clip = mels[:, :, start:start+clip_size]
            pred_starts, pred_ends = model(clip)
            pred_ends = torch.nn.functional.sigmoid(pred_ends).squeeze(-1).squeeze(0)  # Squeeze off the batch and sigmoid dimensions, leaving only the sequence dimension.
            indices = torch.nonzero(pred_ends > cutoff_pred_percent)
            for i in indices:
                i = i.item()
                sentence = mels[0, :, last_detection_start:start+i]
                if sentence.shape[-1] > 400 and sentence.shape[-1] < 1600:
                    save_mel_buffer_to_file(sentence, f'{sentence_number}.npy')
                    wav = vocoder.transform_mel_to_audio(sentence)
                    wavfile.write(f'{sentence_number}.wav', 22050, wav[0].cpu().numpy())
                sentence_number += 1
                last_detection_start = start+i
            start += 4
            if last_detection_start > start:
                start = last_detection_start
