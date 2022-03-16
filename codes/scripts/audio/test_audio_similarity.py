import os

import torch
import torch.nn.functional as F

from data.util import is_wav_file, find_files_of_type
from models.audio.audio_resnet import resnet50
from models.audio.tts.tacotron2.taco_utils import load_wav_to_torch
from scripts.byol.byol_extract_wrapped_model import extract_byol_model_from_state_dict

if __name__ == '__main__':
    window = 48000
    root_path = 'D:\\tmp\\clips'
    paths = find_files_of_type('img', root_path, qualifier=is_wav_file)[0]
    clips = []
    for path in paths:
        clip, sr = load_wav_to_torch(os.path.join(root_path, path))
        if len(clip.shape) > 1:
            clip = clip[:,0]
        clip = clip[:window].unsqueeze(0)
        clip = clip / 32768.0  # Normalize
        #clip = clip + torch.rand_like(clip) * .03  # Noise (this is how the model was trained)
        assert sr == 24000
        clips.append(clip)
    clips = torch.stack(clips, dim=0)

    resnet = resnet50()
    sd = torch.load('../experiments/train_byol_audio_clips/models/8000_generator.pth')
    sd = extract_byol_model_from_state_dict(sd)
    resnet.load_state_dict(sd)
    embedding = resnet(clips, return_pool=True)

    for i, path in enumerate(paths):
        print(f'Using a baseline of {path}..')
        for j, cpath in enumerate(paths):
            if i == j:
                continue
            l2 = F.mse_loss(embedding[j], embedding[i])
            print(f'Compared to {cpath}: {l2}')

