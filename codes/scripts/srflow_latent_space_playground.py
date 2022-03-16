import argparse
import logging
import math
import os
from glob import glob

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm

import utils.options as option

import utils
from data.images.image_corruptor import ImageCorruptor
from trainer.ExtensibleTrainer import ExtensibleTrainer
from utils import util


def image_2_tensor(impath, max_size=None):
    img = Image.open(impath)

    if max_size is not None:
        factor = min(max_size / img.width, max_size / img.height)
        new_size = (int(math.ceil(img.width * factor)), int(math.ceil(img.height * factor)))
        img = img.resize(new_size, Image.LANCZOS)

        '''
        # Useful for setting an image to an exact size.
        h_gap = img.height - desired_size[1]
        w_gap = img.width - desired_size[0]
        assert h_gap >= 0 and w_gap >= 0
        ht = h_gap // 2
        hb = desired_size[1] + ht
        wl = w_gap // 2
        wr = desired_size[1] + wl
        '''

    timg = torchvision.transforms.ToTensor()(img).unsqueeze(0)

    #if desired_size is not None:
    #    timg = timg[:, :3, ht:hb, wl:wr]
    #    assert timg.shape[2] == desired_size[1] and timg.shape[3] == desired_size[0]
    #else:
    # Enforce that the input must have a input dimension that is a factor of 16.
    b, c, h, w = timg.shape
    h = (h // 16) * 16
    w = (w // 16) * 16
    timg = timg[:, :3, :h, :w]

    return timg


def interpolate_lr(hr, scale):
    return F.interpolate(hr, scale_factor=1 / scale, mode="area")


def fetch_latents_for_image(gen, img, scale, lr_infer=interpolate_lr):
    z, _, _ = gen(gt=img,
                  lr=lr_infer(img, scale),
                  epses=[],
                  reverse=False,
                  add_gt_noise=False)
    return z


def fetch_latents_for_images(gen, imgs, scale, lr_infer=interpolate_lr):
    latents = []
    for img in imgs:
        z, _, _ = gen(gt=img,
                      lr=lr_infer(img, scale),
                      epses=[],
                      reverse=False,
                      add_gt_noise=False)
        latents.append(z)
    return latents


def fetch_spatial_metrics_for_latents(latents):
    dt_scales = []
    dt_biases = []
    for i in range(len(latents)):
        latent = torch.stack(latents[i], dim=-1).squeeze(0)
        s = latent.std(dim=[1, 2, 3]).view(1,-1,1,1)
        b = latent.mean(dim=[1, 2, 3]).view(1,-1,1,1)
        dt_scales.append(s)
        dt_biases.append(b)
    return dt_scales, dt_biases


def spatial_norm(latents, exclusion_list=[]):
    nlatents = []
    for i in range(len(latents)):
        latent = latents[i]
        if i in exclusion_list:
            nlatents.append(latent)
        else:
            b, c, h, w = latent.shape
            s = latent.std(dim=[2, 3]).view(1,c,1,1)
            b = latent.mean(dim=[2, 3]).view(1,c,1,1)
            nlatents.append((latents[i] - b) / s)
    return nlatents


def local_norm(latents, exclusion_list=[]):
    nlatents = []
    for i in range(len(latents)):
        latent = latents[i]
        if i in exclusion_list:
            nlatents.append(latent)
        else:
            b, c, h, w = latent.shape
            s = latent.std(dim=[1]).view(1,1,h,w)
            b = latent.mean(dim=[1]).view(1,1,h,w)
            nlatents.append((latents[i] - b) / s)
    return nlatents


# Extracts a rectangle of the same shape as <lat> from <ref> and returns it. This is taken from the center of <ref>
def extract_center_latent(ref, lat):
    _, _, h, w = lat.shape
    _, _, rh, rw = ref.shape
    dw = (rw - w) / 2
    dh = (rh - h) / 2
    return ref[:, :, math.floor(dh):-math.ceil(dh), math.floor(dw):-math.ceil(dw)]


def linear_interpolation(latents1, latents2, proportion):
    return [l1*proportion+l2*(1-proportion) for l1, l2 in zip(latents1, latents2)]


def slerp(latents1, latents2, proportion):
    res = []
    for low, high in zip(latents1, latents2):
        low_norm = low / torch.norm(low, dim=[2,3], keepdim=True)
        high_norm = high / torch.norm(high, dim=[2,3], keepdim=True)
        omega = torch.acos((low_norm * high_norm).sum(1))
        so = torch.sin(omega)
        res.append((torch.sin((1.0 - proportion) * omega) / so).unsqueeze(1) * low + (torch.sin(proportion * omega) / so).unsqueeze(1) * high)
    return res


def create_interpolation_video(gen, lq, output_file, latents1, latents2, steps=10, prefix=''):
    # Outputs a series of images interpolated from [latents1] to [latents2]. image 0 biases towards latents2.
    for i in range(steps):
        proportion = i / (steps-1)
        lats = linear_interpolation(latents1, latents2, proportion)
        hr, _, _ = gen(lr=lq,
                    z=lats[0],
                    reverse=True,
                    epses=lats,
                    add_gt_noise=False)
        torchvision.transforms.ToPILImage()(hr.squeeze(0).cpu())
        torchvision.utils.save_image(hr.cpu(), os.path.join(output_file, "%s_%i.png" % (prefix, i)))
    # Stopped using this because PILs animated gif output is total crap.
    #images[0].save(output_file, save_all=True, append_images=images[1:], duration=80, loop=0)


if __name__ == "__main__":
    #### options
    torch.backends.cudnn.benchmark = True
    srg_analyze = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../options/train_exd_imgsetext_srflow_bigboi_ganbase.yml')
    #parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../../options/train_exd_imgsetext_srflow_bigboi_frompsnr.yml')
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

    model = ExtensibleTrainer(opt)
    gen = model.networks['generator']
    gen.eval()

    mode = "feed_through"  # temperature | restore | latent_transfer | feed_through
    #imgs_to_resample_pattern = "F:\\4k6k\\datasets\\ns_images\\adrianna\\val2\\lr\\*"
    imgs_to_resample_pattern = "F:\\4k6k\\datasets\\ns_images\\adrianna\\analyze\\analyze_xx\\*"
    #imgs_to_resample_pattern = "F:\\4k6k\\datasets\\ns_images\\imagesets\\images-half\\*lanette*"
    scale = 2
    resample_factor = 2  # When != 1, the HR image is upsampled by this factor using a bicubic to get the local latents. E.g. set this to '2' to get 2x upsampling.
    temperature = 1
    output_path = "..\\..\\results\\latent_playground"

    # Data types <- used to perform latent transfer.
    data_path = "F:\\4k6k\\datasets\\ns_images\\imagesets\\images-half"
    #data_type_filters = ["*alexa*", "*lanette*", "*80755*", "*joli_high*"]
    data_type_filters = ["*lanette*"]
    max_size = 1600  # Should be set to 2x the largest single dimension of the input space, otherwise an error will occur.
    max_ref_datatypes = 30  # Only picks this many images from the above data types to sample from.
    interpolation_steps = 30

    with torch.no_grad():
        # Compute latent variables for the reference images.
        if mode == "latent_transfer":
            # Just get the **one** result for each pattern and use that latent.
            dt_imgs = [glob(os.path.join(data_path, p))[-5] for p in data_type_filters]
            dt_transfers = [image_2_tensor(i, max_size) for i in dt_imgs]
            # Downsample the images because they are often just too big to feed through the network (probably needs to be parameterized)
            #for j in range(len(dt_transfers)):
            #    if min(dt_transfers[j].shape[2], dt_transfers[j].shape[3]) > 1600:
            #        dt_transfers[j] = F.interpolate(dt_transfers[j], scale_factor=1 / 2, mode='area')
            corruptor = ImageCorruptor({'fixed_corruptions': ['jpeg-medium', 'gaussian_blur_3']})
            def corrupt_and_downsample(img, scale):
                img = F.interpolate(img, scale_factor=1 / scale, mode="area")
                from data.util import torch2cv, cv2torch
                cvimg = torch2cv(img)
                cvimg = corruptor.corrupt_images([cvimg])[0]
                img = cv2torch(cvimg)
                return img
            dt_latents = [fetch_latents_for_image(gen, i, scale, corrupt_and_downsample) for i in dt_transfers]

        # Fetch the images to resample.
        img_files = glob(imgs_to_resample_pattern)
        #random.shuffle(img_files)
        for im_it, img_file in enumerate(tqdm(img_files)):
            t = image_2_tensor(img_file).to(model.env['device'])
            if resample_factor > 1:
                t = F.interpolate(t, scale_factor=resample_factor, mode="bicubic")
            elif resample_factor < 1:
                t = F.interpolate(t, scale_factor=resample_factor, mode="area")
            # Ensure the input image is a factor of 16.
            _, _, h, w = t.shape
            h = 16 * (h // 16)
            w = 16 * (w // 16)
            t = t[:, :, :h, :w]
            resample_img = t

            # Fetch the latent metrics & latents for each image we are resampling.
            latents = fetch_latents_for_images(gen, [resample_img], scale)[0]

            multiple_latents = False
            if mode == "restore":
                latents = local_norm(spatial_norm(latents))
                #latents = spatial_norm(latents)
                latents = [l * temperature for l in latents]
            elif mode == "feed_through":
                latents = [torch.randn_like(l) * temperature for l in latents]
            elif mode == "latent_transfer":
                dts = []
                for slat in dt_latents:
                    assert slat[0].shape[2] >= latents[0].shape[2]
                    assert slat[0].shape[3] >= latents[0].shape[3]
                    dts.append([extract_center_latent(sl, l) * temperature for l, sl in zip(latents, slat)])
                latents = dts
                multiple_latents = True
            elif mode == "temperature":
                latents = [l * temperature for l in latents]

            # Re-compute each image with the new metrics
            if not multiple_latents:
                lats = [latents]
            else:
                lats = latents
            for j in range(len(lats)):
                path = os.path.join(output_path, "%i_%i" % (im_it, j))
                os.makedirs(path, exist_ok=True)
                torchvision.utils.save_image(resample_img, os.path.join(path, "orig_%i.jpg" % (im_it)))
                create_interpolation_video(gen, F.interpolate(resample_img, scale_factor=1/scale, mode="area"),
                                           path, [torch.zeros_like(l) for l in lats[j]], lats[j], prefix=mode)
