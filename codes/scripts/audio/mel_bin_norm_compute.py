import argparse

import torch
import yaml
from tqdm import tqdm

from data import create_dataset, create_dataloader
from trainer.injectors.base_injectors import TorchMelSpectrogramInjector
from utils.options import Loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file used to train the diffusion model', default='D:\\dlas\\options\\train_dvae_audio_clips.yml')
    parser.add_argument('-key', type=str, help='Key where audio data is stored', default='clip')
    parser.add_argument('-num_batches', type=str, help='Number of batches to collect to compute the norm', default=10)
    args = parser.parse_args()

    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    dopt = opt['datasets']['train']
    dopt['phase'] = 'train'
    dataset, collate = create_dataset(dopt, return_collate=True)
    dataloader = create_dataloader(dataset, dopt, collate_fn=collate, shuffle=True)
    inj = TorchMelSpectrogramInjector({'in': 'wav', 'out': 'mel'},{}).cuda()

    mels = []
    for batch in tqdm(dataloader):
        clip = batch[args.key].cuda()
        mel = inj({'wav': clip})['mel']
        mels.append(mel.mean((0,2)).cpu())
        if len(mels) > args.num_batches:
            break
    mel_norms = torch.stack(mels).mean(0)
    torch.save('mel_norms.pth', mel_norms)