import os

import torch
from tqdm import tqdm

from scripts.audio.gen.speech_synthesis_utils import load_speech_dvae, wav_to_mel

if __name__ == '__main__':
    input_folder = 'C:\\Users\\James\\Downloads\\lex2\\lexfridman_training_mp3'
    output_folder = 'C:\\Users\\James\\Downloads\\lex2\\quantized'

    params = {
        'mode': 'unsupervised_audio',
        'path': [input_folder],
        'cache_path': f'{input_folder}/cache.pth',
        'sampling_rate': 22050,
        'pad_to_samples': 441000,
        'resample_clip': False,
        'extra_samples': 0,
        'phase': 'train',
        'n_workers': 2,
        'batch_size': 64,
    }
    from data import create_dataset, create_dataloader
    os.makedirs(output_folder, exist_ok=True)

    ds = create_dataset(params)
    dl = create_dataloader(ds, params)

    dvae = load_speech_dvae().cuda()
    with torch.no_grad():
        for batch in tqdm(dl):
            audio = batch['clip'].cuda()
            mel = wav_to_mel(audio)
            codes = dvae.get_codebook_indices(mel)
            for i in range(audio.shape[0]):
                c = codes[i, :batch['clip_lengths'][i]//1024+4]  # +4 seems empirically to be a good clipping point - it seems to preserve the termination codes.
                fn = batch['path'][i]
                outp = os.path.join(output_folder, os.path.relpath(fn, input_folder) + ".pth")
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                torch.save(c.tolist(), outp)
