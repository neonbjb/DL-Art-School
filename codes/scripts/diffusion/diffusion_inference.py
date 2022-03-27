import argparse
import os

import torch
from torchvision.io import read_image
from torchvision.utils import save_image

from models.diffusion.gaussian_diffusion import get_named_beta_schedule
from models.diffusion.respace import SpacedDiffusion, space_timesteps
from models.diffusion.unet_diffusion import SuperResModel
from utils.util import ceil_multiple


def load_model():
    model = SuperResModel(image_size=256, in_channels=3, num_corruptions=2, model_channels=192, out_channels=6, num_res_blocks=2,
                      attention_resolutions=[8,16], dropout=0, channel_mult=[1,1,2,2,4,4], num_heads=4, num_heads_upsample=-1,
                      use_scale_shift_norm=True)
    sd = torch.load('../experiments/diffusion_unet_111500.pth')
    model.load_state_dict(sd)
    model.eval()
    return model


def read_and_constrain_image(img_path):
    """
    The input image into the diffusion model must have dimensions that are a multiple of 32. This function adds padding
    to make it so.
    """
    img = 2 * (read_image(img_path) / 255) - 1
    # Get rid of alpha channel if present
    img = img[:3]
    assert img.shape[0] == 3  # Does not support greyscale images anyways.
    _, h, w = img.shape
    dh = ceil_multiple(h, 32)
    dw = ceil_multiple(w, 32)
    return torch.nn.functional.pad(img, (0,dh-h,0,dw-w))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image to repair and super-resolve.')
    parser.add_argument('--blur_correction', type=float, help='Blur correction factor; [0,1]', default=.1)
    parser.add_argument('--jpeg_correction', type=float, help='Compression noise correction factor; [0,1]', default=0)
    parser.add_argument('--sr_factor', type=int, help='Multiplicative amount to super-resolve the image; [1,4]', default=2)
    parser.add_argument('--diffusion_steps', type=int, help='Number of diffusion steps. Lower is faster, higher makes higher quality images. >400 is unnecessary. [0,4000]', default=100)
    parser.add_argument('--output', type=str, help='Where to store output image', default='.')
    parser.add_argument('--device', type=str, help='Device to perform inference on; cpu or cuda', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    model = load_model().to(args.device)
    diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [args.diffusion_steps]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 4000))
    lr_image = read_and_constrain_image(args.image).unsqueeze(0).to(args.device)

    with torch.no_grad():
        output_shape = (1, 3, lr_image.shape[-2]*args.sr_factor, lr_image.shape[-1]*args.sr_factor)
        cfactor = torch.tensor([[args.jpeg_correction, args.blur_correction]], device=args.device, dtype=torch.float)
        hq = diffuser.p_sample_loop(model, output_shape, model_kwargs={'low_res': lr_image, 'corruption_factor': cfactor})
        hq = (hq + 1) / 2
        save_image(hq, os.path.join(args.output, os.path.basename(args.image)))