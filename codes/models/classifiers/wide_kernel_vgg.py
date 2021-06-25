import torch
import torch.nn as nn

from trainer.networks import register_model
from utils.util import opt_get


class WideKernelVgg(nn.Module):
    def __init__(self, nf=64, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            # [64, 128, 128]
            nn.Conv2d(6, nf, 7, 1, 3, bias=True),
            nn.BatchNorm2d(nf, affine=True),
            nn.ReLU(),
            nn.Conv2d(nf, nf, 7, 1, 3, bias=False),
            nn.BatchNorm2d(nf, affine=True),
            nn.ReLU(),
            nn.Conv2d(nf, nf, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf, affine=True),
            nn.ReLU(),
            # [64, 64, 64]
            nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 2, affine=True),
            nn.ReLU(),
            nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2, affine=True),
            nn.ReLU(),
            # [128, 32, 32]
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 4, affine=True),
            nn.ReLU(),
            nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4, affine=True),
            nn.ReLU(),
            # [256, 16, 16]
            nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 8, affine=True),
            nn.ReLU(),
            nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8, affine=True),
            nn.ReLU(),
            # [512, 8, 8]
            nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 8, affine=True),
            nn.ReLU(),
            nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(nf * 8 * 4 * 2, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

        # These normalization constants should be derived experimentally.
        self.log_fft_mean = torch.tensor([-3.5184, -4.071]).view(1,1,1,2)
        self.log_fft_std = torch.tensor([3.1660, 3.8042]).view(1,1,1,2)

    def forward(self, x):
        b,c,h,w = x.shape
        x_c = x.view(c*b, h, w)
        x_c = torch.view_as_real(torch.fft.rfft(x_c))

        # Log-normalize spectrogram
        x_c = (x_c.abs() ** 2).clip(min=1e-8, max=1e16)
        x_c = torch.log(x_c)
        x_c = (x_c - self.log_fft_mean.to(x.device)) / self.log_fft_std.to(x.device)

        # Return to expected input shape (b,c,h,w)
        x_c = x_c.permute(0, 3, 1, 2).reshape(b, c * 2, h, w // 2 + 1)

        return self.net(x_c)


@register_model
def register_wide_kernel_vgg(opt_net, opt):
    """ return a ResNet 18 object
    """
    return WideKernelVgg(**opt_get(opt_net, ['kwargs'], {}))


if __name__ == '__main__':
    vgg = WideKernelVgg()
    vgg(torch.randn(1,3,256,256))