import argparse

import torch
import yaml
from tqdm import tqdm

from data import create_dataset, create_dataloader
from scripts.audio.gen.speech_synthesis_utils import wav_to_univnet_mel
from utils.options import Loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file used to train the diffusion model', default='D:\\dlas\\options\\train_diffusion_tts9.yml')
    parser.add_argument('-key', type=str, help='Key where audio data is stored', default='wav')
    parser.add_argument('-num_batches', type=int, help='Number of batches to collect to compute the norm', default=50000)
    args = parser.parse_args()

    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    dopt = opt['datasets']['train']
    dopt['phase'] = 'train'
    dataset, collate = create_dataset(dopt, return_collate=True)
    dataloader = create_dataloader(dataset, dopt, collate_fn=collate, shuffle=True)

    mel_means = []
    mel_max = -999999999
    mel_min = 999999999
    mel_stds = []
    mel_vars = []
    for batch in tqdm(dataloader):
        if len(mel_means) > args.num_batches:
            break
        clip = batch[args.key].cuda()
        for b in range(clip.shape[0]):
            wav = clip[b].unsqueeze(0)
            wav = wav[:, :, :batch[f'{args.key}_lengths'][b]]
            mel = wav_to_univnet_mel(clip)  # Caution: make sure this isn't already normed.
            mel_means.append(mel.mean((0,2)).cpu())
            mel_max = max(mel.max().item(), mel_max)
            mel_min = min(mel.min().item(), mel_min)
            mel_stds.append(mel.std((0,2)).cpu())
            mel_vars.append(mel.var((0,2)).cpu())
    mel_means = torch.stack(mel_means).mean(0)
    mel_stds = torch.stack(mel_stds).mean(0)
    mel_vars = torch.stack(mel_vars).mean(0)
    torch.save((mel_means,mel_max,mel_min,mel_stds,mel_vars), 'univnet_mel_norms.pth')