import torch
import torchvision
from PIL import Image

def load_img(path):
    im = Image.open(path)
    return torchvision.transforms.ToTensor()(im)

def save_img(t, path):
    torchvision.utils.save_image(t, path)

img = load_img("me.png")
# add zeros to the imaginary component
img = torch.stack([img, torch.zeros_like(img)], dim=-1)
fft = torch.fft(img, signal_ndim=2)
fft_d = torch.zeros_like(fft)
for i in range(-5, 5):
    diag = torch.diagonal(fft, offset=i, dim1=1, dim2=2)
    diag_em = torch.diag_embed(diag, offset=i, dim1=1, dim2=2)
    fft_d += diag_em
resamp_img = torch.ifft(fft_d, signal_ndim=2)[:, :, :, 0]
save_img(resamp_img, "resampled.png")