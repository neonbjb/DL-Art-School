import os
import os.path as osp
import logging
import random
import argparse

from PIL import Image
from scipy.io import wavfile
from torchvision.transforms import ToTensor

import utils
import utils.options as option
import utils.util as util
from data.audio.unsupervised_audio_dataset import load_audio
from trainer.ExtensibleTrainer import ExtensibleTrainer
import torch
import numpy as np

# A rough copy of test.py that "surfs" along a set of random noise priors to show the affect of gaussian noise on the results.


def forward_pass(model, data, output_dir, spacing, audio_mode):
    with torch.no_grad():
        model.feed_data(data, 0)
        model.test()

    visuals = model.get_current_visuals()['rlt'].cpu()
    img_path = data['GT_path'][0]
    img_name = osp.splitext(osp.basename(img_path))[0]
    sr_img = visuals[0]

    # save images
    suffixes = [f'_{int(spacing)}']
    for suffix in suffixes:
        if audio_mode:
            save_img_path = osp.join(output_dir, img_name + suffix + '.wav')
            wavfile.write(osp.join(output_dir, save_img_path), 11025, sr_img[0].cpu().numpy())
        else:
            save_img_path = osp.join(output_dir, img_name + suffix + '.png')
            util.save_img(util.tensor2img(sr_img), save_img_path)


def load_image(path, audio_mode):
    # Load test image
    if audio_mode:
        im = load_audio(path, 22050).unsqueeze(0)
    else:
        im = ToTensor()(Image.open(path)) * 2 - 1
        _, h, w = im.shape
        if h % 2 == 1:
            im = im[:,1:,:]
            h = h-1
        if w % 2 == 1:
            im = im[:,:,1:]
            w = w-1
        dh, dw = (h - 32 * (h // 32)) // 2, (w - 32 * (w // 32)) // 2
        if dh > 0:
            im = im[:,dh:-dh]
        if dw > 0:
            im = im[:,:,dw:-dw]
        im = im[:3].unsqueeze(0)
    return im


if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(5555)
    random.seed(5555)
    np.random.seed(5555)

    #### options
    audio_mode = True  # Whether to render audio or images.
    torch.backends.cudnn.benchmark = True
    want_metrics = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../options/test_diffusion_vocoder_dvae.yml')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    utils.util.loaded_options = opt

    util.mkdirs(
        (path for key, path in opt['path'].items()
         if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    im = load_image(opt['image'], audio_mode)
    correction_factors = util.opt_get(opt, ['correction_factor'], None)
    if 'ref_images' in opt.keys():
        refs = [load_image(r, audio_mode) for r in opt['ref_images']]
        #min_len = min(r.shape[1] for r in refs)
        min_len = opt['ref_images_len']
        refs = [r[:, :min_len] for r in refs]
        refs = torch.stack(refs, dim=1)
    else:
        refs = torch.empty((1,1))

    #opt['steps']['generator']['injectors']['visual_debug']['zero_noise'] = False
    model = ExtensibleTrainer(opt)
    results_dir = osp.join(opt['path']['results_root'], os.path.basename(opt['image']))
    util.mkdir(results_dir)
    for i in range(10):
        if audio_mode:
            data = {
                'clip': im.to('cuda'),
                'alt_clips': refs.to('cuda'),
                'num_alt_clips': torch.tensor([refs.shape[1]], dtype=torch.int32, device='cuda'),
                'GT_path': opt['image'],
                'resampled_clip': refs[:, 0].to('cuda')
            }
        else:
            data = {
                'hq': im.to('cuda'),
                'corruption_entropy': torch.tensor([correction_factors], device='cuda',
                                                   dtype=torch.float),
                'GT_path': opt['image']
            }
        forward_pass(model, data, results_dir, i, audio_mode)
