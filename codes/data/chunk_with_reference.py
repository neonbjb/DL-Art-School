import os.path as osp
from data import util
import torch
import numpy as np

# Iterable that reads all the images in a directory that contains a reference image, tile images and center coordinates.
from utils.util import opt_get


class ChunkWithReference:
    def __init__(self, opt, path):
        self.path = path.path
        self.tiles, _ = util.find_files_of_type('img', self.path)
        self.need_metadata = opt_get(opt, ['strict'], False) or opt_get(opt, ['needs_metadata'], False)
        self.need_ref = opt_get(opt, ['need_ref'], False)
        if 'ignore_first' in opt.keys():
            self.tiles = self.tiles[opt['ignore_first']:]

    # Odd failures occur at times. Rather than crashing, report the error and just return zeros.
    def read_image_or_get_zero(self, img_path):
        img = util.read_img(None, img_path, rgb=True)
        if img is None:
            return np.zeros(128, 128, 3)
        return img

    def __getitem__(self, item):
        tile = self.read_image_or_get_zero(self.tiles[item])
        if self.need_ref and osp.exists(osp.join(self.path, "ref.jpg")):
            tile_id = int(osp.splitext(osp.basename(self.tiles[item]))[0])
            ref = self.read_image_or_get_zero(osp.join(self.path, "ref.jpg"))
            if self.need_metadata:
                centers = torch.load(osp.join(self.path, "centers.pt"))
                if tile_id in centers.keys():
                    center, tile_width = centers[tile_id]
                else:
                    print("Could not find the given tile id in the accompanying centers.pt. This generally means that "
                          "centers.pt was overwritten at some point e.g. by duplicate data. If you don't care about tile "
                          "centers, consider passing strict=false to the dataset options. (Note: you must re-build your"
                          "caches for this setting change to take effect.)")
                    raise FileNotFoundError(tile_id, self.tiles[item])
            else:
                center = torch.tensor([128, 128], dtype=torch.long)
                tile_width = 256
            mask = np.full(tile.shape[:2] + (1,), fill_value=.1, dtype=tile.dtype)
            mask[center[0] - tile_width // 2:center[0] + tile_width // 2, center[1] - tile_width // 2:center[1] + tile_width // 2] = 1
        else:
            ref = np.zeros_like(tile)
            mask = np.zeros(tile.shape[:2] + (1,))
            center = (0,0)

        return tile, ref, center, mask, self.tiles[item]

    def __len__(self):
        return len(self.tiles)
