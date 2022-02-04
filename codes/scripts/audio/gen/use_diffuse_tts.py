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
    conditioning_clips = {
        # Male
        'simmons': 'Y:\\clips\\books1\\754_Dan Simmons - The Rise Of Endymion 356 of 450\\00026.wav',
        'carlin': 'Y:\\clips\\books1\\12_dchha13 Bubonic Nukes\\00097.wav',
        'entangled': 'Y:\\clips\\books1\\3857_25_The_Entangled_Bank__000000000\\00123.wav',
        'snowden': 'Y:\\clips\\books1\\7658_Edward_Snowden_-_Permanent_Record__000000004\\00027.wav',
        # Female
        'the_doctor': 'Y:\\clips\\books2\\37062___The_Doctor__000000003\\00206.wav',
        'puppy': 'Y:\\clips\\books2\\17830___3_Puppy_Kisses__000000002\\00046.wav',
        'adrift': 'Y:\\clips\\books2\\5608_Gear__W_Michael_-_Donovan_1-5_(2018-2021)_(book_4_Gear__W_Michael_-_Donovan_5_-_Adrift_(2021)_Gear__W_Michael_-_Adrift_(Donovan_5)_—_82__000000000\\00019.wav',
    }

    provided_codes = [
        # but facts within easy reach of any one who cares to know them go to say that the greater abstenence of women is in some part
        # due to an imperative conventionality and this conventionality is in a general way strongest were the patriarchal tradition
        # the tradition that the woman is a chattel has retained its hold in greatest vigor
        # 3570/5694/3570_5694_000008_000001.wav
        [0, 0, 24, 0, 16, 0, 6, 0, 4, 0, 0, 0, 0, 0, 20, 0, 7, 0, 0, 19, 19, 0, 0, 6, 0, 0, 12, 12, 0, 4, 4, 0, 18, 18,
         0, 10, 0, 6, 11, 11, 10, 10, 9, 9, 4, 4, 4, 5, 5, 0, 7, 0, 0, 0, 0, 12, 0, 22, 22, 0, 4, 4, 0, 13, 13, 5, 0, 7,
         7, 0, 0, 19, 11, 0, 4, 4, 8, 20, 4, 4, 4, 7, 0, 9, 9, 0, 22, 4, 4, 0, 8, 0, 9, 5, 4, 4, 18, 11, 11, 8, 4, 4, 0,
         0, 0, 19, 19, 7, 0, 0, 13, 5, 5, 0, 12, 12, 4, 4, 6, 6, 8, 8, 4, 4, 0, 26, 9, 9, 8, 0, 18, 0, 0, 4, 4, 6, 6,
         11, 5, 0, 17, 17, 0, 0, 4, 4, 4, 4, 0, 0, 0, 21, 0, 8, 0, 0, 0, 0, 4, 4, 6, 6, 8, 0, 4, 4, 0, 0, 12, 0, 7, 7,
         0, 0, 22, 0, 4, 4, 6, 11, 11, 7, 6, 6, 4, 4, 6, 11, 5, 4, 4, 4, 0, 21, 0, 13, 5, 5, 7, 7, 0, 0, 6, 6, 5, 0, 13,
         0, 4, 4, 0, 7, 0, 0, 0, 24, 0, 0, 12, 12, 0, 0, 6, 0, 5, 0, 0, 9, 9, 0, 5, 0, 9, 0, 0, 19, 5, 5, 4, 4, 8, 20,
         20, 4, 4, 4, 4, 0, 18, 18, 8, 0, 0, 0, 17, 0, 5, 0, 9, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 10, 0, 0, 12, 12, 4, 4, 0,
         10, 0, 9, 0, 4, 4, 0, 0, 12, 0, 0, 8, 0, 17, 5, 5, 4, 4, 0, 0, 0, 23, 23, 0, 7, 0, 13, 0, 0, 0, 6, 0, 4, 0, 0,
         0, 0, 14, 0, 16, 16, 0, 0, 5, 0, 4, 4, 0, 6, 8, 0, 4, 4, 7, 9, 4, 4, 4, 0, 10, 10, 17, 0, 0, 0, 23, 0, 5, 0, 0,
         13, 13, 0, 7, 0, 0, 6, 6, 0, 10, 0, 25, 5, 5, 4, 4, 0, 0, 0, 19, 19, 8, 8, 9, 0, 0, 0, 0, 0, 25, 0, 5, 0, 9, 0,
         0, 0, 6, 6, 10, 8, 8, 0, 9, 0, 0, 0, 7, 0, 0, 15, 0, 10, 0, 0, 0, 0, 6, 6, 0, 0, 22, 0, 0, 0, 4, 4, 4, 4, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         7, 0, 9, 14, 0, 4, 0, 0, 6, 11, 10, 0, 0, 0, 12, 0, 4, 4, 0, 19, 19, 8, 9, 9, 0, 0, 25, 0, 5, 0, 9, 0, 0, 6, 6,
         10, 8, 8, 9, 9, 0, 0, 7, 0, 0, 15, 0, 10, 0, 0, 0, 0, 6, 0, 22, 22, 0, 4, 4, 0, 0, 10, 0, 0, 0, 0, 12, 12, 0,
         0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 10, 0, 9, 4, 4, 4, 7, 4, 4, 4, 0, 21, 0, 5, 0, 9, 0, 5, 5, 13, 13, 7, 0, 15,
         15, 0, 0, 4, 4, 0, 18, 18, 0, 7, 0, 0, 22, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 12, 12, 0, 0, 0, 6, 6, 13, 13, 8, 0, 0, 9, 9, 0, 21, 0, 0, 5, 5, 0, 0, 0, 12, 12, 0, 0, 6,
         0, 0, 0, 4, 4, 0, 0, 0, 18, 0, 5, 0, 13, 0, 5, 4, 4, 6, 11, 5, 0, 4, 4, 23, 23, 7, 7, 0, 0, 0, 6, 0, 13, 13,
         10, 10, 0, 0, 0, 0, 7, 13, 13, 0, 19, 11, 11, 0, 0, 7, 15, 15, 0, 0, 4, 4, 0, 6, 13, 13, 7, 7, 0, 0, 0, 14, 10,
         10, 0, 0, 0, 0, 0, 6, 10, 10, 8, 8, 9, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 11, 5, 0, 4, 4, 0, 6, 13, 13, 7, 7, 0, 0, 0, 14, 10, 10, 0, 0, 0, 6, 10, 10,
         8, 9, 9, 0, 0, 4, 4, 0, 6, 11, 7, 0, 6, 4, 4, 6, 11, 5, 4, 4, 4, 18, 18, 8, 0, 0, 17, 7, 0, 9, 0, 4, 10, 0, 0,
         12, 12, 4, 4, 4, 7, 4, 4, 0, 0, 0, 19, 11, 0, 7, 0, 6, 0, 0, 0, 6, 0, 5, 0, 15, 15, 0, 0, 0, 4, 4, 4, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 7, 0, 0, 0, 12, 0, 0, 4, 4, 0, 13, 5, 5, 0, 0, 0, 0, 6, 6, 0, 0,
         7, 10, 10, 0, 9, 0, 5, 0, 14, 4, 4, 4, 0, 10, 0, 0, 0, 6, 0, 0, 0, 0, 0, 12, 0, 4, 4, 0, 0, 0, 11, 0, 0, 8, 0,
         0, 0, 15, 0, 0, 14, 0, 4, 4, 4, 0, 10, 0, 9, 4, 4, 4, 4, 4, 0, 21, 0, 13, 5, 5, 7, 7, 0, 0, 6, 0, 5, 0, 0, 12,
         0, 6, 0, 4, 0, 0, 25, 10, 0, 0, 0, 21, 0, 8, 0, 0, 13, 13, 0, 0, 4, 4, 4, 4, 0, 0, 0],
        # the competitor with whom the entertainer wishes to institute a comparison is by this method made to serve as a means to the end
        # 3570/5694/3570_5694_000011_000005.wav
        [0, 0, 6, 11, 5, 0, 4, 0, 19, 19, 8, 17, 0, 0, 0, 0, 23, 0, 5, 5, 0, 0, 6, 6, 10, 10, 0, 0, 6, 6, 0, 8, 0, 13,
         13, 0, 4, 4, 18, 18, 10, 0, 6, 11, 11, 4, 4, 4, 0, 0, 18, 18, 11, 0, 8, 0, 0, 0, 0, 17, 0, 0, 4, 0, 6, 11, 5,
         0, 4, 4, 0, 5, 9, 9, 0, 6, 5, 5, 13, 13, 0, 0, 6, 6, 0, 7, 0, 10, 0, 9, 0, 0, 5, 0, 13, 4, 4, 0, 18, 10, 10, 0,
         0, 12, 11, 11, 0, 5, 0, 0, 0, 12, 0, 0, 4, 4, 0, 0, 6, 6, 8, 0, 0, 4, 4, 4, 0, 10, 9, 9, 0, 0, 0, 0, 12, 0, 0,
         6, 0, 10, 0, 0, 0, 6, 0, 16, 16, 0, 6, 5, 0, 4, 4, 7, 4, 4, 19, 19, 8, 0, 17, 0, 0, 0, 0, 0, 23, 0, 0, 7, 0, 0,
         0, 13, 0, 10, 0, 0, 0, 0, 0, 12, 0, 0, 8, 0, 9, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 24, 0, 22, 0, 4, 4,
         0, 6, 11, 10, 0, 0, 0, 12, 0, 0, 4, 4, 0, 0, 17, 5, 5, 0, 0, 0, 6, 11, 11, 8, 0, 0, 14, 14, 0, 0, 4, 4, 4, 4,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 17, 7, 0, 0, 0, 0, 14, 5, 0, 4, 4, 6, 8, 4,
         4, 0, 0, 0, 12, 12, 0, 5, 5, 0, 13, 13, 0, 25, 5, 4, 4, 7, 0, 12, 4, 4, 4, 7, 4, 4, 0, 17, 5, 0, 0, 7, 0, 0, 9,
         0, 0, 0, 0, 12, 0, 4, 4, 0, 6, 0, 8, 0, 4, 4, 6, 11, 5, 4, 4, 4, 0, 0, 5, 0, 9, 9, 0, 0, 0, 0, 14, 0, 0, 4, 4,
         4, 4, 4, 0, 0],
        # the livery becomes obnoxious to nearly all who are required to wear it
        # 3570/5694/3570_5694_000014_000021.wav
        [0, 0, 6, 11, 5, 0, 0, 4, 4, 0, 15, 10, 10, 0, 0, 25, 5, 0, 13, 13, 0, 22, 0, 0, 4, 0, 24, 24, 5, 0, 0, 0, 19,
         19, 0, 8, 0, 17, 5, 5, 0, 12, 0, 4, 4, 4, 0, 8, 0, 0, 24, 0, 0, 0, 9, 9, 0, 8, 0, 0, 0, 0, 0, 28, 0, 0, 0, 10,
         0, 8, 16, 0, 12, 12, 12, 0, 4, 0, 6, 6, 8, 0, 4, 4, 0, 9, 5, 0, 7, 7, 13, 0, 0, 15, 22, 22, 4, 4, 0, 0, 0, 0,
         0, 0, 0, 7, 0, 15, 0, 0, 15, 0, 4, 4, 4, 18, 11, 11, 8, 0, 4, 4, 0, 7, 0, 13, 5, 4, 4, 13, 13, 5, 0, 0, 0, 30,
         30, 16, 0, 0, 10, 0, 0, 0, 13, 5, 0, 14, 4, 4, 6, 6, 8, 0, 4, 4, 18, 18, 5, 5, 7, 7, 13, 13, 0, 4, 4, 0, 10, 0,
         0, 0, 0, 6, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0],
        # in the nature of things luxuries and the comforts of life belong to the leisure class
        # 3570/5694/3570_5694_000006_000007.wav
        [0, 0, 0, 0, 0, 10, 9, 0, 4, 4, 6, 11, 5, 4, 4, 4, 9, 9, 7, 7, 0, 0, 0, 0, 0, 0, 6, 0, 16, 16, 13, 13, 5, 0, 4, 4, 8, 0, 20, 4, 4, 4, 0, 6, 0, 11, 10, 0, 9, 0, 21, 0, 0, 0, 12, 12, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 15, 0, 16, 16, 0, 0, 28, 0, 0, 0, 16, 16, 0, 13, 13, 0, 10, 0, 5, 0, 0, 0, 12, 0, 0, 4, 4, 4, 0, 0, 7, 0, 9, 0, 14, 4, 4, 6, 11, 5, 4, 4, 0, 0, 19, 0, 8, 17, 17, 0, 0, 0, 0, 0, 20, 0, 8, 0, 13, 0, 6, 0, 12, 4, 4, 8, 0, 20, 4, 4, 4, 0, 0, 15, 0, 10, 10, 0, 0, 0, 20, 5, 0, 4, 4, 0, 0, 24, 5, 0, 0, 0, 15, 8, 0, 9, 0, 21, 0, 0, 0, 4, 4, 6, 8, 4, 4, 4, 6, 11, 5, 4, 4, 15, 15, 5, 10, 0, 0, 12, 0, 16, 13, 5, 5, 4, 4, 0, 19, 0, 15, 15, 0, 0, 7, 0, 0, 12, 12, 0, 0, 0, 12, 12, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0],
        # from arcaic times down through all the length of the patriarchal regime it has been the office of the women to
        # prepare and administer these luxuries and it has been the perquisite of the men of gentle birth and breeding
        # to consume them
        # 3570/5694/3570_5694_000007_000003.wav
        [0, 0, 0, 0, 0, 0, 20, 13, 8, 0, 17, 0, 4, 4, 0, 7, 0, 13, 0, 0, 0, 0, 0, 19, 0, 0, 0, 7, 0, 0, 0, 0, 10, 0, 19, 0, 0, 0, 4, 4, 0, 0, 0, 0, 6, 0, 0, 0, 10, 0, 0, 17, 5, 0, 0, 0, 12, 0, 4, 0, 0, 0, 0, 14, 0, 0, 8, 0, 18, 0, 0, 0, 9, 0, 0, 0, 0, 4, 4, 0, 0, 0, 6, 11, 13, 8, 0, 16, 21, 21, 11, 0, 4, 4, 7, 0, 15, 0, 15, 15, 4, 4, 6, 11, 5, 5, 4, 4, 0, 15, 0, 5, 0, 0, 9, 9, 0, 21, 0, 0, 6, 11, 0, 4, 4, 8, 8, 20, 4, 4, 4, 6, 11, 5, 4, 4, 0, 0, 0, 23, 0, 7, 7, 0, 0, 0, 0, 0, 6, 6, 13, 13, 13, 10, 0, 0, 0, 0, 0, 7, 13, 13, 0, 19, 11, 11, 11, 0, 0, 7, 15, 15, 0, 4, 4, 4, 13, 13, 5, 0, 0, 0, 0, 21, 21, 0, 0, 10, 0, 0, 0, 0, 17, 5, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 6, 4, 4, 0, 0, 11, 7, 7, 0, 0, 12, 0, 4, 4, 0, 24, 5, 0, 0, 5, 5, 9, 0, 4, 6, 6, 11, 5, 4, 4, 0, 0, 8, 0, 20, 0, 0, 0, 20, 0, 10, 0, 0, 0, 19, 5, 0, 4, 4, 8, 0, 20, 4, 4, 6, 11, 5, 4, 4, 4, 18, 8, 0, 0, 0, 17, 5, 0, 9, 9, 0, 0, 4, 4, 0, 6, 6, 8, 0, 0, 4, 4, 0, 23, 23, 13, 5, 5, 0, 0, 0, 0, 23, 23, 0, 7, 0, 0, 0, 13, 5, 0, 0, 0, 4, 4, 0, 7, 0, 9, 14, 0, 4, 4, 0, 0, 7, 0, 14, 0, 0, 0, 17, 17, 10, 0, 9, 0, 10, 10, 0, 0, 12, 12, 0, 0, 0, 6, 0, 5, 13, 13, 0, 0, 0, 0, 4, 4, 4, 6, 11, 11, 5, 0, 0, 0, 12, 5, 5, 4, 4, 15, 15, 0, 16, 0, 0, 0, 28, 0, 0, 0, 16, 0, 0, 13, 13, 10, 0, 5, 5, 0, 0, 12, 12, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 9, 0, 14, 4, 4, 10, 0, 6, 4, 4, 0, 11, 11, 7, 0, 0, 0, 12, 0, 4, 4, 0, 0, 0, 0, 24, 5, 0, 0, 5, 5, 9, 9, 4, 4, 4, 6, 11, 5, 4, 4, 0, 0, 0, 23, 0, 5, 0, 13, 0, 0, 0, 0, 0, 30, 30, 16, 10, 10, 0, 0, 0, 12, 0, 10, 0, 0, 6, 5, 0, 4, 4, 8, 20, 0, 4, 4, 6, 11, 5, 4, 4, 0, 17, 5, 0, 0, 0, 9, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 20, 4, 4, 4, 0, 0, 21, 0, 5, 5, 0, 9, 9, 0, 0, 0, 6, 0, 15, 0, 5, 0, 4, 0, 0, 0, 24, 0, 10, 0, 13, 0, 0, 0, 0, 6, 11, 0, 0, 4, 0, 0, 7, 0, 9, 14, 14, 4, 4, 4, 0, 0, 24, 13, 5, 0, 0, 0, 5, 0, 0, 14, 10, 0, 9, 21, 21, 0, 4, 4, 0, 6, 8, 0, 4, 4, 0, 19, 8, 0, 9, 0, 0, 0, 0, 0, 0, 0, 12, 0, 16, 0, 17, 5, 0, 0, 4, 4, 6, 11, 5, 0, 17, 0, 4, 4, 4, 4, 0, 0],
        # yes it is perfection she declared
        # 1284/1180/1284_1180_000036_000000.wav
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 4, 4, 4, 4, 0, 0, 10, 0, 6, 0, 4, 4, 0, 0, 10, 0, 0, 0, 0, 0, 12, 0, 4, 4, 0, 0, 0, 23, 0, 5, 0, 13, 13, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 5, 0, 0, 0, 19, 0, 0, 6, 6, 0, 10, 0, 8, 0, 9, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 12, 11, 11, 5, 0, 4, 4, 0, 14, 0, 5, 0, 0, 0, 0, 19, 15, 15, 0, 0, 7, 0, 0, 0, 13, 0, 5, 0, 14, 4, 4, 4, 4, 0, 0, 0],
        # then it must be somewhere in the blue forest
        # 1284/1180/1284_1180_000016_000002.wav
        [0, 0, 0, 6, 11, 5, 0, 9, 0, 4, 4, 10, 6, 4, 4, 0, 17, 17, 16, 0, 0, 12, 0, 6, 4, 4, 0, 24, 5, 5, 0, 0, 4, 4, 0, 0, 12, 12, 0, 8, 0, 0, 17, 5, 5, 0, 0, 18, 18, 11, 5, 0, 13, 13, 5, 0, 4, 4, 10, 9, 4, 4, 6, 11, 5, 4, 4, 0, 24, 15, 15, 16, 16, 0, 5, 5, 0, 0, 4, 4, 0, 0, 0, 20, 8, 8, 8, 0, 0, 0, 13, 13, 0, 5, 5, 0, 0, 0, 0, 0, 12, 12, 0, 0, 6, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0],
        # happy youth that is ready to pack its valus and start for cathay on an hour's notice
        # 4970/29093/4970_29093_000044_000002.wav
        [0, 0, 0, 0, 11, 0, 7, 23, 0, 0, 0, 0, 23, 0, 22, 22, 0, 0, 0, 4, 4, 0, 0, 22, 8, 8, 16, 16, 0, 0, 0, 6, 6, 11, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 11, 7, 6, 0, 4, 4, 10, 0, 0, 12, 0, 4, 0, 13, 13, 5, 0, 7, 0, 0, 14, 22, 0, 0, 0, 4, 0, 6, 0, 8, 4, 4, 0, 0, 0, 0, 0, 0, 23, 0, 7, 0, 0, 19, 0, 0, 26, 4, 4, 4, 10, 0, 6, 0, 12, 4, 4, 0, 0, 0, 25, 0, 7, 0, 0, 0, 15, 0, 0, 16, 0, 0, 0, 0, 12, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 9, 0, 14, 4, 4, 0, 12, 12, 0, 6, 0, 7, 0, 13, 0, 0, 0, 6, 0, 0, 4, 4, 0, 0, 0, 0, 20, 8, 0, 13, 0, 4, 4, 4, 0, 0, 19, 0, 7, 7, 0, 0, 0, 0, 0, 6, 11, 0, 0, 7, 0, 0, 0, 22, 0, 0, 0, 0, 0, 4, 4, 0, 0, 8, 0, 9, 0, 4, 4, 7, 9, 4, 4, 4, 0, 0, 0, 11, 8, 8, 16, 0, 0, 13, 13, 0, 0, 0, 27, 0, 12, 0, 4, 4, 0, 9, 8, 8, 0, 0, 0, 0, 6, 10, 0, 0, 0, 0, 0, 19, 5, 5, 0, 0, 4, 4, 4, 4, 4, 0],
        # well then i must make some suggestions to you
        # 1580/141084/1580_141084_000057_000000.wav
        [0, 0, 0, 0, 0, 0, 0, 18, 0, 5, 0, 15, 0, 0, 15, 15, 4, 4, 0, 0, 6, 11, 5, 0, 0, 0, 9, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 10, 0, 4, 4, 0, 17, 0, 16, 0, 0, 12, 0, 6, 0, 4, 4, 0, 17, 17, 7, 0, 26, 5, 5, 4, 4, 0, 12, 12, 8, 8, 17, 17, 5, 0, 4, 4, 4, 12, 12, 16, 0, 21, 0, 0, 0, 0, 21, 21, 0, 5, 0, 0, 0, 12, 0, 0, 0, 6, 6, 0, 10, 0, 8, 8, 9, 0, 0, 0, 0, 0, 0, 12, 0, 0, 4, 4, 0, 0, 6, 0, 8, 0, 4, 4, 4, 0, 0, 22, 22, 0, 8, 16, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0],
        # some others too big cotton county
        # 1995/1826/1995_1826_000010_000002.wav
        [0, 0, 0, 0, 12, 0, 8, 0, 17, 5, 4, 4, 0, 8, 0, 0, 6, 11, 5, 0, 13, 13, 0, 0, 12, 0, 4, 4, 0, 0, 6, 0, 8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 10, 0, 0, 0, 0, 21, 0, 0, 4, 4, 4, 0, 0, 0, 19, 0, 8, 0, 6, 6, 0, 0, 0, 6, 8, 0, 9, 9, 0, 0, 4, 0, 0, 0, 0, 19, 8, 8, 16, 0, 9, 9, 0, 0, 6, 6, 0, 0, 22, 0, 0, 0, 0, 4, 4, 0, 0, 0],
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file used to train the diffusion model', default='X:\\dlas\\experiments\\train_diffusion_tts5_medium.yml')
    parser.add_argument('-diffusion_model_name', type=str, help='Name of the diffusion model in opt.', default='generator')
    parser.add_argument('-diffusion_model_path', type=str, help='Path to saved model weights', default='X:\\dlas\\experiments\\train_diffusion_tts5_medium\\models\\68500_generator_ema.pth')
    # -cond "Y:\libritts/train-clean-100/103/1241/103_1241_000017_000001.wav"
    parser.add_argument('-cond', type=str, help='Type of conditioning voice', default='simmons')
    parser.add_argument('-diffusion_steps', type=int, help='Number of diffusion steps to perform to create the generate. Lower steps reduces quality, but >40 is generally pretty good.', default=100)
    parser.add_argument('-diffusion_schedule', type=str, help='Type of diffusion schedule that was used', default='cosine')
    parser.add_argument('-output_path', type=str, help='Where to store outputs.', default='../results/use_diffuse_tts')
    parser.add_argument('-sample_rate', type=int, help='Model sample rate', default=5500)
    parser.add_argument('-cond_sample_rate', type=int, help='Conditioning sample rate', default=5500)
    parser.add_argument('-device', type=str, help='Device to run on', default='cuda')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    print("Loading Diffusion Model..")
    diffusion = load_model_from_config(args.opt, args.diffusion_model_name, also_load_savepoint=False,
                                       load_path=args.diffusion_model_path, device=args.device)
    diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=args.diffusion_steps, schedule=args.diffusion_schedule)
    aligned_codes_compression_factor = args.sample_rate * 221 // 11025
    cond = load_audio(conditioning_clips[args.cond], args.cond_sample_rate).to(args.device)
    if cond.shape[-1] > 88000:
        cond = cond[:,:88000]
    torchaudio.save(os.path.join(args.output_path, 'cond.wav'), cond.cpu(), args.sample_rate)

    for p, code in enumerate(provided_codes):
        print("Loading data..")
        aligned_codes = torch.tensor(code).to(args.device)

        with torch.no_grad():
            print("Performing inference..")
            diffusion.eval()
            output_shape = (1, 1, ceil_multiple(aligned_codes.shape[-1]*aligned_codes_compression_factor, 2048))

            output = diffuser.p_sample_loop(diffusion, output_shape, noise=torch.zeros(output_shape, device=args.device),
                                            model_kwargs={'tokens': aligned_codes.unsqueeze(0),
                                            'conditioning_input': cond.unsqueeze(0)})
            torchaudio.save(os.path.join(args.output_path, f'{p}_output_mean.wav'), output.cpu().squeeze(0), args.sample_rate)

            for k in range(2):
                output = diffuser.p_sample_loop(diffusion, output_shape, model_kwargs={'tokens': aligned_codes.unsqueeze(0),
                                                                                       'conditioning_input': cond.unsqueeze(0)})

                torchaudio.save(os.path.join(args.output_path, f'{p}_output_{k}.wav'), output.cpu().squeeze(0), args.sample_rate)
