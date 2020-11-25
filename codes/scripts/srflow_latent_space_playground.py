import argparse
import logging
import math
import os
import random
from glob import glob

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm

import utils.options as option

import utils
from data import create_dataset, create_dataloader
from data.image_corruptor import ImageCorruptor
from models.ExtensibleTrainer import ExtensibleTrainer
from utils import util


def image_2_tensor(impath, desired_size):
    img = Image.open(impath)

    if desired_size is not None:
        factor = max(desired_size[0] / img.width, desired_size[1] / img.height)
        new_size = (int(math.ceil(img.width * factor)), int(math.ceil(img.height * factor)))
        img = img.resize(new_size, Image.BICUBIC)

        h_gap = img.height - desired_size[1]
        w_gap = img.width - desired_size[0]
        assert h_gap >= 0 and w_gap >= 0
        ht = h_gap // 2
        hb = desired_size[1] + ht
        wl = w_gap // 2
        wr = desired_size[1] + wl

    timg = torchvision.transforms.ToTensor()(img).unsqueeze(0)

    if desired_size is not None:
        timg = timg[:, :3, ht:hb, wl:wr]
        assert timg.shape[2] == desired_size[1] and timg.shape[3] == desired_size[0]
    else:
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
    for img in tqdm(imgs):
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


def spatial_norm(latents):
    nlatents = []
    for i in range(len(latents)):
        latent = latents[i]
        b, c, h, w = latent.shape
        s = latent.std(dim=[2, 3]).view(1,c,1,1)
        b = latent.mean(dim=[2, 3]).view(1,c,1,1)
        nlatents.append((latents[i] - b) / s)
    return nlatents


def local_norm(latents):
    nlatents = []
    for i in range(len(latents)):
        latent = latents[i]
        b, c, h, w = latent.shape
        s = latent.std(dim=[1]).view(1,1,h,w)
        b = latent.mean(dim=[1]).view(1,1,h,w)
        nlatents.append((latents[i] - b) / s)
    return nlatents


if __name__ == "__main__":
    #### options
    torch.backends.cudnn.benchmark = True
    srg_analyze = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../../experiments/train_exd_imgset_srflow/train_exd_imgset_srflow.yml')
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

    mode = "latent_transfer"
    imgs_to_resample_pattern = "F:\\4k6k\\datasets\\ns_images\\adrianna\\val2\\lr\\*"
    desired_size = None # (640,640)  # <- Required when doing style transfer.
    scale = 2
    resample_factor = 2  # When != 1, the HR image is upsampled by this factor using a bicubic to get the local latents.
    temperature = .65
    output_path = "E:\\4k6k\\mmsr\\results\\latent_playground"

    # Data types <- used to perform latent transfer.
    data_path = "F:\\4k6k\\datasets\\ns_images\\imagesets\\images-half"
    data_type_filters = ["*alexa*", "*lanette*", "*80755*", "*x-art-1912*", "*joli_high*", "*stacy-cruz*"]
    #data_type_filters = ["*lanette*"]
    max_ref_datatypes = 30  # Only picks this many images from the above data types to sample from.
    interpolation_steps = 30

    with torch.no_grad():
        # Fetch the images to resample.
        resample_imgs = []
        img_files = glob(imgs_to_resample_pattern)
        for i, img_file in enumerate(img_files):
            #if i > 5:
            #    break
            t = image_2_tensor(img_file, desired_size).to(model.env['device'])
            if resample_factor != 1:
                t = F.interpolate(t, scale_factor=resample_factor, mode="bicubic")
            resample_imgs.append(t)

        # Fetch the latent metrics & latents for each image we are resampling.
        latents = fetch_latents_for_images(gen, resample_imgs, scale)

        multiple_latents = False
        if mode == "restore":
            for i, latent_set in enumerate(latents):
                latents[i] = local_norm(spatial_norm(latent_set))
                latents[i] = [l * temperature for l in latents[i]]
        elif mode == "feed_through":
            latents = [torch.randn_like(l) * temperature for l in latents[i]]
        elif mode == "latent_transfer":
            # Just get the **one** result for each pattern and use that latent.
            dt_imgs = [glob(os.path.join(data_path, p))[-5] for p in data_type_filters]
            dt_transfers = [image_2_tensor(i, desired_size) for i in dt_imgs]
            # Downsample the images because they are often just too big to feed through the network (probably needs to be parameterized)
            for j in range(len(dt_transfers)):
                if max(dt_transfers[j].shape[2], dt_transfers[j].shape[3]) > 2000:
                    dt_transfers[j] = F.interpolate(dt_transfers[j], scale_factor=1/2, mode='area')
            corruptor = ImageCorruptor({'fixed_corruptions':['jpeg-low', 'gaussian_blur_5']})

            def corrupt_and_downsample(img, scale):
                img = F.interpolate(img, scale_factor=1/scale, mode="area")
                from data.util import torch2cv, cv2torch
                cvimg = torch2cv(img)
                cvimg = corruptor.corrupt_images([cvimg])[0]
                img = cv2torch(cvimg)
                torchvision.utils.save_image(img, "corrupted_lq_%i.png" % (random.randint(0,100),))
                return img

            dt_latents = [fetch_latents_for_image(gen, i, scale, corrupt_and_downsample) for i in dt_transfers]
            tlatents = []
            for lat in latents:
                dts = []
                for slat in dt_latents:
                    assert slat[0].shape[2] >= lat[0].shape[2]
                    assert slat[0].shape[3] >= lat[0].shape[3]
                    dts.append([sl[:,:,:l.shape[2],:l.shape[3]] * temperature for l, sl in zip(lat, slat)])
                tlatents.append(dts)
            latents = tlatents
            multiple_latents = True

        # Re-compute each image with the new metrics
        for i, img in enumerate(resample_imgs):
            if not multiple_latents:
                lats = [latents[i]]
            else:
                lats = latents[i]
            for j in range(len(lats)):
                hr, _ = gen(lr=F.interpolate(img, scale_factor=1/scale, mode="area"),
                         z=lats[j][0],
                         reverse=True,
                         epses=lats[j],
                         add_gt_noise=False)
                os.makedirs(os.path.join(output_path), exist_ok=True)
                torchvision.utils.save_image(hr, os.path.join(output_path, "%i_%i.png" % (i,j)))
