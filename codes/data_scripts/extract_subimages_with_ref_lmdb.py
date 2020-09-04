"""A multi-thread tool to crop large images to sub-images for faster IO."""
import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
import data.util as data_util  # noqa: E402
import lmdb
import pyarrow
import torch.utils.data as data
from tqdm import tqdm


def main():
    mode = 'single'  # single (one input folder) | pair (extract corresponding GT and LR pairs)
    split_img = False
    opt = {}
    opt['n_thread'] = 12
    opt['compression_level'] = 90  # JPEG compression quality rating.
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.
    if mode == 'single':
        opt['input_folder'] = 'F:\\4k6k\\datasets\\ns_images\\imagesets\\images'
        opt['save_folder'] = 'F:\\4k6k\\datasets\\ns_images\\imagesets\\lmdb_with_ref'
        opt['crop_sz'] = 512  # the size of each sub-image
        opt['step'] = 128  # step of the sliding crop window
        opt['thres_sz'] = 128  # size threshold
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


class LmdbWriter:
    def __init__(self, lmdb_path, max_mem_size=30*1024*1024*1024, write_freq=5000):
        self.db = lmdb.open(lmdb_path, subdir=True,
                       map_size=max_mem_size, readonly=False,
                       meminit=False, map_async=True)
        self.txn = self.db.begin(write=True)
        self.ref_id = 0
        self.tile_ids = {}
        self.writes = 0
        self.write_freq = write_freq
        self.keys = []

    # Writes the given reference image to the db and returns its ID.
    def write_reference_image(self, ref_img):
        id = self.ref_id
        self.ref_id += 1
        self.write_image(id, ref_img[0], ref_img[1])
        return id

    # Writes a tile image to the db given a reference image and returns its ID.
    def write_tile_image(self, ref_id, tile_image):
        next_tile_id = 0 if ref_id not in self.tile_ids.keys() else self.tile_ids[ref_id]
        self.tile_ids[ref_id] = next_tile_id+1
        full_id = "%i_%i" % (ref_id, next_tile_id)
        self.write_image(full_id, tile_image[0], tile_image[1])
        self.keys.append(full_id)
        return full_id

    # Writes an image directly to the db with the given reference image and center point.
    def write_image(self, id, img, center_point):
        self.txn.put(u'{}'.format(id).encode('ascii'), pyarrow.serialize(img).to_buffer(), pyarrow.serialize(center_point).to_buffer())
        self.writes += 1
        if self.writes % self.write_freq == 0:
            self.txn.commit()
            self.txn = self.db.begin(write=True)

    def close(self):
        self.txn.commit()
        with self.db.begin(write=True) as txn:
            txn.put(b'__keys__', pyarrow.serialize(self.keys).to_buffer())
            txn.put(b'__len__', pyarrow.serialize(len(self.keys)).to_buffer())
        self.db.sync()
        self.db.close()


class TiledDataset(data.Dataset):
    def __init__(self, opt, split_mode=False):
        self.split_mode = split_mode
        self.opt = opt
        input_folder = opt['input_folder']
        self.images = data_util._get_paths_from_images(input_folder)

    def __getitem__(self, index):
        if self.split_mode:
            return self.get(index, True, True).extend(self.get(index, True, False))
        else:
            return self.get(index, False, False)

    def get(self, index, split_mode, left_img):
        path = self.images[index]
        crop_sz = self.opt['crop_sz']
        step = self.opt['step']
        thres_sz = self.opt['thres_sz']
        only_resize = self.opt['only_resize']
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # We must convert the image into a square. Crop the image so that only the center is left, since this is often
        # the most salient part of the image.
        h, w, c = img.shape
        dim = min(h, w)
        img = img[(h - dim) // 2:dim + (h - dim) // 2, (w - dim) // 2:dim + (w - dim) // 2, :]

        h, w, c = img.shape
        # Uncomment to filter any image that doesnt meet a threshold size.
        if min(h,w) < 1024:
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
        resize_factor = self.opt['resize_final_img'] if 'resize_final_img' in self.opt.keys() else 1
        dsize = (int(crop_sz * resize_factor), int(crop_sz * resize_factor))
        # Reference image should always be first.
        results = [(cv2.resize(img, dsize, interpolation=cv2.INTER_AREA), (-1,-1))]
        for x in h_space:
            for y in w_space:
                index += 1
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
                center_point = (x + crop_sz // 2, y + crop_sz // 2)
                crop_img = np.ascontiguousarray(crop_img)
                if 'resize_final_img' in self.opt.keys():
                    # Resize too.
                    resize_factor = self.opt['resize_final_img']
                    center_point = (int(center_point[0] * resize_factor), int(center_point[1] * resize_factor))
                    crop_img = cv2.resize(crop_img, dsize, interpolation=cv2.INTER_AREA)
                success, buffer = cv2.imencode(".jpg", crop_img, [cv2.IMWRITE_JPEG_QUALITY, self.opt['compression_level']])
                assert success
                results.append((buffer, center_point))
        return results

    def __len__(self):
        return len(self.images)


def identity(x):
    return x

def extract_single(opt, split_img=False):
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    lmdb = LmdbWriter(save_folder)

    dataset = TiledDataset(opt, split_img)
    dataloader = data.DataLoader(dataset, num_workers=opt['n_thread'], collate_fn=identity)
    tq = tqdm(dataloader)
    for imgs in tq:
        if imgs is None or len(imgs) <= 1:
            continue
        ref_id = lmdb.write_reference_image(imgs[0])
        for tile in imgs[1:]:
            lmdb.write_tile_image(ref_id, tile)
    lmdb.close()


if __name__ == '__main__':
    main()
