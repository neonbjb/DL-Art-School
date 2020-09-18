import os.path as osp
from data import util
import torch

# Iterable that reads all the images in a directory that contains a reference image, tile images and center coordinates.
class ChunkWithReference:
    def __init__(self, opt, path):
        self.opt = opt
        self.path = path
        self.ref = None  # This is loaded on the fly.
        self.cache_ref = opt['cache_ref'] if 'cache_ref' in opt.keys() else True
        self.tiles = util.get_image_paths('img', path)

    def __getitem__(self, item):
        if self.cache_ref:
            if self.ref is None:
                self.ref = util.read_img(None, osp.join(self.path, "ref.jpg"))
                self.centers = torch.load(osp.join(self.path, "centers.pt"))
            ref = self.ref
            centers = self.centers
        else:
            self.ref = util.read_img(None, osp.join(self.path, "ref.jpg"))
            self.centers = torch.load(osp.join(self.path, "centers.pt"))

        return self.tiles[item], ref, centers[item], path

    def __len__(self):
        return len(self.tiles)