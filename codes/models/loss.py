import torch
import torch.nn as nn
import numpy as np
from utils.fdpl_util import extract_patches_2d, dct_2d
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
        loss = self.loss(input, target_label)
        return loss


# Frequency Domain Perceptual Loss, from https://github.com/sdv4/FDPL
# Utilizes pre-computed perceptual_weights. To generate these from your dataset, see data_scripts/compute_fdpl_perceptual_weights.py
# In practice, per the paper, these precomputed weights can generally be used across broad image classes (e.g. all photographs).
class FDPLLoss(nn.Module):
    """
        Loss function taking the MSE between the 2D DCT coefficients
        of predicted and target images or an image channel.
        DCT coefficients are computed for each 8x8 block of the image

        Important note about this loss: Since it operates in the frequency domain, precision is highly important.
        It works on FP64 numbers and will fail if you attempt to use it with AMP. Recommend you split this loss
        off from the rest of amp.scale_loss().
    """

    def __init__(self, dataset_diff_means_file, device):
        """
            dataset_diff_means (torch.tensor): Pre-computed frequency-domain mean differences between LR and HR images.
        """
        # These values are derived from the JPEG standard.
        qt_Y = torch.tensor([[16, 11, 10, 16, 24, 40, 51, 61],
                             [12, 12, 14, 19, 26, 58, 60, 55],
                             [14, 13, 16, 24, 40, 57, 69, 56],
                             [14, 17, 22, 29, 51, 87, 80, 62],
                             [18, 22, 37, 56, 68, 109, 103, 77],
                             [24, 35, 55, 64, 81, 104, 113, 92],
                             [49, 64, 78, 87, 103, 121, 120, 101],
                             [72, 92, 95, 98, 112, 100, 103, 99]],
                            dtype=torch.double,
                            device=device,
                            requires_grad=False)
        qt_C = torch.tensor([[17, 18, 24, 47, 99, 99, 99, 99],
                             [18, 21, 26, 66, 99, 99, 99, 99],
                             [24, 26, 56, 99, 99, 99, 99, 99],
                             [47, 66, 99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99, 99, 99]],
                            dtype=torch.double,
                            device=device,
                            requires_grad=False)
        """
            Reasoning behind this perceptual weight matrix: JPEG gives as a model of frequencies that are important 
            for human perception. In that model, lower frequencies are more important than higher frequencies. Because
            of this, the higher frequencies are the first to go during compression. As compression increases, the affect
            spreads to the lower frequencies, which degrades perceptual quality. But when the lower frequencies are
            preserved, JPEG does an excellent job of compression without a noticeable loss of quality.
            In super resolution, we already have the low frequencies. In fact that is really all we have in the low
            resolution images.
            As evidenced by the diff_means matrix above, what is lost in the SR images is the mid-range frequencies; 
            those across and towards the centre of the diagonal. We can bias our model to recover these frequencies
            by having our loss function prioritize these coefficients, with priority determined by the magnitude of 
            relative change between the low-res and high-res images. But we can take this further and into a true 
            preceptual loss by further prioritizing DCT coefficients by the importance that has been assigned to them 
            by the JPEG quantization table. That is how the table below is created. 

            The problem is that we don't know if the JPEG model is optimal. So there is room for qualitative evaluation
            of the quantization table values. We can further our perspective weights deleting each in turn for a small
            set of images and evaluating the resulting change in percieved quality. I can do this on my own to start and 
            if it works, I can do a small user study to determine this. 
        """
        diff_means = torch.tensor(torch.load(dataset_diff_means_file), device=device)
        perceptual_weights = torch.stack([(torch.ones_like(qt_Y, device=device) / qt_Y),
                                          (torch.ones_like(qt_C, device=device) / qt_C),
                                          (torch.ones_like(qt_C, device=device) / qt_C)])
        perceptual_weights = perceptual_weights * diff_means
        self.perceptual_weights = perceptual_weights / torch.mean(perceptual_weights)
        super(FDPLLoss, self).__init__()

    def forward(self, predictions, targets):
        """
            Args:
                predictions (torch.tensor): output of an image transformation model.
                    shape: batch_size x 3 x H x W
                targets (torch.tensor): ground truth images corresponding to outputs
                    shape: batch_size x 3 x H x W
                criterion (torch.nn.MSELoss): object used to calculate MSE

            Returns:
                loss (float): MSE between predicted and ground truth 2D DCT coefficients
        """
        # transition to fp64 and then convert to YCC color space.
        predictions = rgb2ycbcr(predictions.double())
        targets = rgb2ycbcr(targets.double())

        # get DCT coefficients of ground truth patches
        patches = extract_patches_2d(img=targets, patch_shape=(8, 8), batch_first=True)
        ground_truth_dct = dct_2d(patches, norm='ortho')

        # get DCT coefficients of transformed images
        patches = extract_patches_2d(img=predictions, patch_shape=(8, 8), batch_first=True)
        outputs_dct = dct_2d(patches, norm='ortho')
        loss = torch.sum(((outputs_dct - ground_truth_dct).pow(2)) * self.perceptual_weights)
        return loss