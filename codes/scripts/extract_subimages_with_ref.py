"""A multi-thread tool to crop large images to sub-images for faster IO."""
import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
import data.util as data_util  # noqa: E402
import torch.utils.data as data
from tqdm import tqdm
import torch


def main():
    split_img = False
    opt = {}
    opt['n_thread'] = 8
    opt['compression_level'] = 95  # JPEG compression quality rating.
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

    opt['dest'] = 'file'
    opt['input_folder'] = 'E:\\4k6k\\datasets\\ns_images\\imagesets\\imageset_1024_square_with_new'
    opt['save_folder'] = 'E:\\4k6k\\datasets\\ns_images\\imagesets\\256_only_humans_masked_pt2'
    opt['crop_sz'] = [256, 512]  # the size of each sub-image
    opt['step'] = [256, 512]  # step of the sliding crop window
    opt['exclusions'] = [[],[]] # image names matching these terms wont be included in the processing.
    opt['thres_sz'] = 129  # size threshold
    opt['resize_final_img'] = [1, .5]
    opt['only_resize'] = False
    opt['vertical_split'] = False
    opt['use_masking'] = True
    opt['mask_path'] = 'E:\\4k6k\\datasets\\ns_images\\imagesets\\imageset_1024_square_with_new_masks'
    opt['input_image_max_size_before_being_halved'] = 5500  # As described, images larger than this dimensional size will be halved before anything else is done.
                                                            # This helps prevent images from cameras with "false-megapixels" from polluting the dataset.
                                                            # False-megapixel=lots of noise at ultra-high res.

    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))

    if opt['dest'] == 'lmdb':
        writer = LmdbWriter(save_folder)
    else:
        writer = FileWriter(save_folder)

    extract_single(opt, writer)


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
    def write_reference_image(self, ref_img, _):
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


class FileWriter:
    def __init__(self, folder):
        self.folder = folder
        self.next_unique_id = 0
        self.ref_center_points = {}   # Maps ref_img basename to a dict of image IDs:center points
        self.ref_ids_to_names = {}

    def get_next_unique_id(self):
        id = self.next_unique_id
        self.next_unique_id += 1
        return id

    def save_image(self, ref_path, img_name, img):
        save_path = osp.join(self.folder, ref_path)
        os.makedirs(save_path, exist_ok=True)
        f = open(osp.join(save_path, img_name), "wb")
        f.write(img)
        f.close()

    # Writes the given reference image to the db and returns its ID.
    def write_reference_image(self, ref_img, path):
        ref_img, _, _ = ref_img  # Encoded with a center point, which is irrelevant for the reference image.
        img_name = osp.basename(path).replace(".jpg", "").replace(".png", "")
        self.ref_center_points[img_name] = {}
        self.save_image(img_name, "ref.jpg", ref_img)
        id = self.get_next_unique_id()
        self.ref_ids_to_names[id] = img_name
        return id

    # Writes a tile image to the db given a reference image and returns its ID.
    def write_tile_image(self, ref_id, tile_image):
        id = self.get_next_unique_id()
        ref_name = self.ref_ids_to_names[ref_id]
        img, center, tile_sz = tile_image
        self.ref_center_points[ref_name][id] = center, tile_sz
        self.save_image(ref_name, "%08i.jpg" % (id,), img)
        return id

    def flush(self):
        for ref_name, cps in self.ref_center_points.items():
            torch.save(cps, osp.join(self.folder, ref_name, "centers.pt"))
        self.ref_center_points = {}

    def close(self):
        self.flush()

class TiledDataset(data.Dataset):
    def __init__(self, opt):
        self.split_mode = opt['vertical_split']
        self.opt = opt
        input_folder = opt['input_folder']
        self.images = data_util._get_paths_from_images(input_folder)

    def __getitem__(self, index):
        if self.split_mode:
            return (self.get(index, True, True), self.get(index, True, False))
        else:
            # Wrap in a tuple to align with split mode.
            return (self.get(index, False, False), None)

    def get_for_scale(self, img, mask, crop_sz, step, resize_factor, ref_resize_factor):
        thres_sz = self.opt['thres_sz']
        h, w, c = img.shape

        if crop_sz > h:
            return []

        h_space = np.arange(0, h - crop_sz + 1, step)
        if h - (h_space[-1] + crop_sz) > thres_sz:
            h_space = np.append(h_space, h - crop_sz)
        w_space = np.arange(0, w - crop_sz + 1, step)
        if w - (w_space[-1] + crop_sz) > thres_sz:
            w_space = np.append(w_space, w - crop_sz)

        index = 0
        tile_dim = int(crop_sz * resize_factor)
        dsize = (tile_dim, tile_dim)
        results = []
        for x in h_space:
            for y in w_space:
                index += 1
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
                if mask is not None:
                    def mask_map(inp):
                        mask_factor = 256 / (crop_sz * ref_resize_factor)
                        return int(inp * mask_factor)
                    crop_mask = mask[mask_map(x):mask_map(x+crop_sz),
                                mask_map(y):mask_map(y+crop_sz),
                                :]
                    if crop_mask.mean() < 255 / 2:  # If at least 50% of the image isn't made up of the type of pixels we want to process, ignore this tile.
                        continue
                # Center point needs to be resized by ref_resize_factor - since it is relative to the reference image.
                center_point = (int((x + crop_sz // 2) // ref_resize_factor), int((y + crop_sz // 2) // ref_resize_factor))
                crop_img = np.ascontiguousarray(crop_img)
                if 'resize_final_img' in self.opt.keys():
                    crop_img = cv2.resize(crop_img, dsize, interpolation=cv2.INTER_AREA)
                success, buffer = cv2.imencode(".jpg", crop_img, [cv2.IMWRITE_JPEG_QUALITY, self.opt['compression_level']])
                assert success
                results.append((buffer, center_point, int(crop_sz // ref_resize_factor)))
        return results

    def get(self, index, split_mode, left_img):
        path = self.images[index]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or len(img.shape) == 2:
            return None

        mask = cv2.imread(os.path.join(self.opt['mask_path'], os.path.basename(path) + ".png"), cv2.IMREAD_UNCHANGED) if self.opt['use_masking'] else None

        h, w, c = img.shape

        if max(h,w) > self.opt['input_image_max_size_before_being_halved']:
            h = h // 2
            w = w // 2
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            #print("Resizing to ", img.shape)

        # Uncomment to filter any image that doesnt meet a threshold size.
        if min(h,w) < 512:
            return None
        # Greyscale not supported.
        if len(img.shape) == 2:
            return None

        # Handle splitting the image if needed.
        left = 0
        right = w
        if split_mode:
            if left_img:
                left = 0
                right = w//2
            else:
                left = w//2
                right = w
        img = img[:, left:right]

        # We must convert the image into a square.
        dim = min(h, w)
        if split_mode:
            # Crop the image towards the center, which makes more sense in split mode.
            if left_img:
                img = img[-dim:, -dim:, :]
            else:
                img = img[:dim, :dim, :]
        else:
            # Crop the image so that only the center is left, since this is often the most salient part of the image.
            img = img[(h - dim) // 2:dim + (h - dim) // 2, (w - dim) // 2:dim + (w - dim) // 2, :]

        h, w, c = img.shape

        tile_dim = int(self.opt['crop_sz'][0] * self.opt['resize_final_img'][0])
        dsize = (tile_dim, tile_dim)
        ref_resize_factor = h / tile_dim

        # Reference image should always be first entry in results.
        ref_img = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
        success, ref_buffer = cv2.imencode(".jpg", ref_img, [cv2.IMWRITE_JPEG_QUALITY, self.opt['compression_level']])
        assert success
        results = [(ref_buffer, (-1,-1), (-1,-1))]

        for crop_sz, exclusions, resize_factor, step in zip(self.opt['crop_sz'], self.opt['exclusions'], self.opt['resize_final_img'], self.opt['step']):
            excluded = False
            for exc in exclusions:
                if exc in path:
                    excluded = True
                    break
            if excluded:
                continue
            results.extend(self.get_for_scale(img, mask, crop_sz, step, resize_factor, ref_resize_factor))
        return results, path

    def __len__(self):
        return len(self.images)


def identity(x):
    return x

def extract_single(opt, writer):
    dataset = TiledDataset(opt)
    dataloader = data.DataLoader(dataset, num_workers=opt['n_thread'], collate_fn=identity)
    tq = tqdm(dataloader)
    i = 0
    for spl_imgs in tq:
        if spl_imgs is None:
            continue
        spl_imgs = spl_imgs[0]
        for imgs, lbl in zip(list(spl_imgs), ['left', 'right']):
            if imgs is None:
                continue
            imgs, path = imgs
            if imgs is None or len(imgs) <= 1:
                continue
            path = f'{path}_{lbl}_{i}'
            i += 1
            ref_id = writer.write_reference_image(imgs[0], path)
            for tile in imgs[1:]:
                writer.write_tile_image(ref_id, tile)
            writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
