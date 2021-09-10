import math

import torch
import torch.nn.functional as F
from torch import nn
from torch import istft

from .unet import UNet
from .util import tf2pytorch


def load_ckpt(model, ckpt):
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            target_shape = state_dict[k].shape
            assert target_shape == v.shape
            state_dict.update({k: torch.from_numpy(v)})
        else:
            print('Ignore ', k)

    model.load_state_dict(state_dict)
    return model


def pad_and_partition(tensor, T):
    """
    pads zero and partition tensor into segments of length T

    Args:
        tensor(Tensor): BxCxFxL

    Returns:
        tensor of size (B*[L/T] x C x F x T)
    """
    old_size = tensor.size(3)
    new_size = math.ceil(old_size/T) * T
    tensor = F.pad(tensor, [0, new_size - old_size])
    [b, c, t, f] = tensor.shape
    split = new_size // T
    return torch.cat(torch.split(tensor, T, dim=3), dim=0)


class Estimator(nn.Module):
    def __init__(self, num_instrumments, checkpoint_path):
        super(Estimator, self).__init__()

        # stft config
        self.F = 1024
        self.T = 512
        self.win_length = 4096
        self.hop_length = 1024
        self.win = torch.hann_window(self.win_length)

        ckpts = tf2pytorch(checkpoint_path, num_instrumments)

        # filter
        self.instruments = nn.ModuleList()
        for i in range(num_instrumments):
            print('Loading model for instrumment {}'.format(i))
            net = UNet(2)
            ckpt = ckpts[i]
            net = load_ckpt(net, ckpt)
            net.eval()  # change mode to eval
            self.instruments.append(net)

    def compute_stft(self, wav):
        """
        Computes stft feature from wav

        Args:
            wav (Tensor): B x L
        """

        stft = torch.stft(
            wav, self.win_length, hop_length=self.hop_length, window=self.win.to(wav.device))

        # only keep freqs smaller than self.F
        stft = stft[:, :self.F, :, :]
        real = stft[:, :, :, 0]
        im = stft[:, :, :, 1]
        mag = torch.sqrt(real ** 2 + im ** 2)

        return stft, mag

    def inverse_stft(self, stft):
        """Inverses stft to wave form"""

        pad = self.win_length // 2 + 1 - stft.size(1)
        stft = F.pad(stft, (0, 0, 0, 0, 0, pad))
        wav = istft(stft, self.win_length, hop_length=self.hop_length,
                    window=self.win.to(stft.device))
        return wav.detach()

    def separate(self, wav):
        """
        Separates stereo wav into different tracks corresponding to different instruments

        Args:
            wav (tensor): B x L
        """

        # stft - B X F x L x 2
        # stft_mag - B X F x L
        stft, stft_mag = self.compute_stft(wav)

        L = stft.size(2)

        stft_mag = stft_mag.unsqueeze(1).repeat(1,2,1,1)  # B x 2 x F x T
        stft_mag = pad_and_partition(stft_mag, self.T)  # B x 2 x F x T
        stft_mag = stft_mag.transpose(2, 3)  # B x 2 x T x F

        # compute instruments' mask
        masks = []
        for net in self.instruments:
            mask = net(stft_mag)
            masks.append(mask)

        # compute denominator
        mask_sum = sum([m ** 2 for m in masks])
        mask_sum += 1e-10

        wavs = []
        for mask in masks:
            mask = (mask ** 2 + 1e-10/2)/(mask_sum)
            mask = mask.transpose(2, 3)  # B x 2 X F x T

            mask = torch.cat(
                torch.split(mask, 1, dim=0), dim=3)

            mask = mask[:,0,:,:L].unsqueeze(-1)  # 2 x F x L x 1
            stft_masked = stft * mask
            wav_masked = self.inverse_stft(stft_masked)

            wavs.append(wav_masked)

        return wavs