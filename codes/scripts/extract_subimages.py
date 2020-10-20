"""A multi-thread tool to crop large images to sub-images for faster IO."""
import os
import os.path as osp
import sys
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils.util import ProgressBar  # noqa: E402
import data.util as data_util  # noqa: E402


def main():
    mode = 'single'  # single (one input folder) | pair (extract corresponding GT and LR pairs)
    split_img = False
    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 90
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.
    if mode == 'single':
        full_multiplier = .25
        opt['input_folder'] = 'F:\\4k6k\\datasets\\images\\fullvideo\\full_images'
        opt['save_folder'] = 'F:\\4k6k\\datasets\\images\\fullvideo\\256_tiled'
        opt['crop_sz'] = int(256 * full_multiplier) # the size of each sub-image
        opt['step'] = int(128 * full_multiplier)  # step of the sliding crop window
        opt['thres_sz'] = int(64 * full_multiplier)  # size threshold
        opt['image_minimum_size_threshold'] = int(1024 * full_multiplier)  # Minimum size of input image in height dim. Images under this size will not be processed.
        opt['resize_final_img'] = .5
        opt['only_resize'] = False
        extract_single(opt, split_img)
    elif mode == 'pair':
        GT_folder = '../../datasets/div2k/DIV2K_train_HR'
        LR_folder = '../../datasets/div2k/DIV2K_train_LR_bicubic/X4'
        save_GT_folder = '../../datasets/div2k/DIV2K800_sub'
        save_LR_folder = '../../datasets/div2k/DIV2K800_sub_bicLRx4'
        scale_ratio = 4
        crop_sz = 480  # the size of each sub-image (GT)
        step = 240  # step of the sliding crop window (GT)
        thres_sz = 48  # size threshold
        ########################################################################
        # check that all the GT and LR images have correct scale ratio
        img_GT_list = data_util._get_paths_from_images(GT_folder)
        img_LR_list = data_util._get_paths_from_images(LR_folder)
        assert len(img_GT_list) == len(img_LR_list), 'different length of GT_folder and LR_folder.'
        for path_GT, path_LR in zip(img_GT_list, img_LR_list):
            img_GT = Image.open(path_GT)
            img_LR = Image.open(path_LR)
            w_GT, h_GT = img_GT.size
            w_LR, h_LR = img_LR.size
            assert w_GT / w_LR == scale_ratio, 'GT width [{:d}] is not {:d}X as LR weight [{:d}] for {:s}.'.format(  # noqa: E501
                w_GT, scale_ratio, w_LR, path_GT)
            assert w_GT / w_LR == scale_ratio, 'GT width [{:d}] is not {:d}X as LR weight [{:d}] for {:s}.'.format(  # noqa: E501
                w_GT, scale_ratio, w_LR, path_GT)
        # check crop size, step and threshold size
        assert crop_sz % scale_ratio == 0, 'crop size is not {:d}X multiplication.'.format(
            scale_ratio)
        assert step % scale_ratio == 0, 'step is not {:d}X multiplication.'.format(scale_ratio)
        assert thres_sz % scale_ratio == 0, 'thres_sz is not {:d}X multiplication.'.format(
            scale_ratio)
        print('process GT...')
        opt['input_folder'] = GT_folder
        opt['save_folder'] = save_GT_folder
        opt['crop_sz'] = crop_sz
        opt['step'] = step
        opt['thres_sz'] = thres_sz
        extract_single(opt)
        print('process LR...')
        opt['input_folder'] = LR_folder
        opt['save_folder'] = save_LR_folder
        opt['crop_sz'] = crop_sz // scale_ratio
        opt['step'] = step // scale_ratio
        opt['thres_sz'] = thres_sz // scale_ratio
        extract_single(opt)
        assert len(data_util._get_paths_from_images(save_GT_folder)) == len(
            data_util._get_paths_from_images(
                save_LR_folder)), 'different length of save_GT_folder and save_LR_folder.'
    else:
        raise ValueError('Wrong mode.')

def extract_single(opt, split_img=False):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    img_list = data_util._get_paths_from_images(input_folder)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(opt['n_thread']) if opt['n_thread'] >= 1 else None
    for path in img_list:
        # If this fails, change it and the imwrite below to the write extension.
        assert ".jpg" in path
        if pool:
            if split_img:
                pool.apply_async(worker, args=(path, opt, True, False), callback=update)
                pool.apply_async(worker, args=(path, opt, True, True), callback=update)
            else:
                pool.apply_async(worker, args=(path, opt), callback=update)
        else:
            assert not split_img
            worker(path, opt)
    pool.close()
    pool.join()
    print('All subprocesses done.')


def worker(path, opt, split_mode=False, left_img=True):
    crop_sz = opt['crop_sz']
    step = opt['step']
    thres_sz = opt['thres_sz']
    only_resize = opt['only_resize']
    img_name = osp.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
       
    # Uncomment to filter any image that doesnt meet a threshold size.
    if min(h,w) < opt['image_minimum_size_threshold']:
        return
    left = 0
    right = w
    if split_mode:
        if left_img:
            left = 0
            right = int(w/2)
        else:
            left = int(w/2)
            right = w
        w = int(w/2)
    img = img[:, left:right]

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    dsize = None
    if only_resize:
        dsize = (crop_sz, crop_sz)
        if h < w:
            h_space = [0]
            w_space = [(w - h) // 2]
            crop_sz = h
        else:
            h_space = [(h - w) // 2]
            w_space = [0]
            crop_sz = w

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            if 'resize_final_img' in opt.keys():
                # Resize too.
                resize_factor = opt['resize_final_img']
                if dsize is None:
                    dsize = (int(crop_img.shape[0] * resize_factor), int(crop_img.shape[1] * resize_factor))
                crop_img = cv2.resize(crop_img, dsize, interpolation = cv2.INTER_AREA)
            cv2.imwrite(
                osp.join(opt['save_folder'],
                         img_name.replace('.jpg', '_l{:05d}_s{:03d}.jpg'.format(left, index))), crop_img,
                [cv2.IMWRITE_JPEG_QUALITY, opt['compression_level']])
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
