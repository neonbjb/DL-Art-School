import os.path as osp
from data import util
import torch
import numpy as np

# Iterable that reads all the images in a directory that contains a reference image, tile images and center coordinates.
class ChunkWithReference:
    def __init__(self, opt, path):
        self.opt = opt
        self.path = path.path
        self.ref = None  # This is loaded on the fly.
        self.cache_ref = opt['cache_ref'] if 'cache_ref' in opt.keys() else True
        self.tiles, _ = util.get_image_paths('img', path)
        self.centers = None

    def __getitem__(self, item):
        # Load centers on the fly and always cache.
        if self.centers is None:
            self.centers = torch.load(osp.join(self.path, "centers.pt"))
        if self.cache_ref:
            if self.ref is None:
                self.ref = util.read_img(None, osp.join(self.path, "ref.jpg"), rgb=True)
                self.centers = torch.load(osp.join(self.path, "centers.pt"))
            ref = self.ref
        else:
            self.ref = util.read_img(None, osp.join(self.path, "ref.jpg"), rgb=True)
        tile = util.read_img(None, self.tiles[item], rgb=True)
        tile_id = int(osp.splitext(osp.basename(self.tiles[item]))[0])
        center, tile_width = self.centers[tile_id]
        mask = np.full(tile.shape[:2] + (1,), fill_value=.1, dtype=tile.dtype)
        mask[center[0] - tile_width // 2:center[0] + tile_width // 2, center[1] - tile_width // 2:center[1] + tile_width // 2] = 1

        return tile, ref, center, mask, self.tiles[item]

    def __len__(self):
        return len(self.tiles)
