import torchvision.utils

from utils.music_utils import music2mel, music2cqt
from utils.util import load_audio

if __name__ == '__main__':
    clip = load_audio('Y:\\split\\yt-music-eval\\00001.wav', 22050)
    mel = music2mel(clip)
    cqt = music2cqt(clip)
    torchvision.utils.save_image((mel.unsqueeze(1) + 1) / 2, 'mel.png')
    torchvision.utils.save_image((cqt.unsqueeze(1) + 1) / 2, 'cqt.png')
