import torch
import torch.nn as nn
import numpy as np
from utils.colors import rgb2ycbcr


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


class ZeroSpreadLoss(nn.Module):
    def __init__(self):
        super(ZeroSpreadLoss, self).__init__()

    def forward(self, x, _):
        return 2 * torch.nn.functional.sigmoid(1 / torch.abs(torch.mean(x))) - 1


# Define GAN loss: [vanilla | lsgan]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type in ['gan', 'ragan', 'pixgan', 'pixgan_fea', 'crossgan', 'crossgan_lrref']:
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'max_spread':
            self.loss = ZeroSpreadLoss()
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        if self.gan_type in ['pixgan', 'pixgan_fea', 'crossgan', 'crossgan_lrref'] and not isinstance(target_is_real, bool):
            target_label = target_is_real
        else:
            target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input.float(), target_label.float())
        return loss
