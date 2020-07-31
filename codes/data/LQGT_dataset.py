import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
from PIL import Image, ImageOps
from io import BytesIO
import torchvision.transforms.functional as F


class LQGTDataset(data.Dataset):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    If only GT images are provided, generate LQ images on-the-fly.
    """

    def get_lq_path(self, i):
        which_lq = random.randint(0, len(self.paths_LQ)-1)
        return self.paths_LQ[which_lq][i]

    def __init__(self, opt):
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.paths_PIX, self.sizes_PIX = None, None
        self.paths_GAN, self.sizes_GAN = None, None
        self.LQ_env, self.GT_env, self.PIX_env = None, None, None  # environments for lmdbs
        self.force_multiple = self.opt['force_multiple'] if 'force_multiple' in self.opt.keys() else 1

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'], opt['dataroot_GT_weights'])
        if 'dataroot_LQ' in opt.keys():
            self.paths_LQ = []
            if isinstance(opt['dataroot_LQ'], list):
                # Multiple LQ data sources can be given, in case there are multiple ways of corrupting a source image and
                # we want the model to learn them all.
                for dr_lq in opt['dataroot_LQ']:
                    lq_path, self.sizes_LQ = util.get_image_paths(self.data_type, dr_lq)
                    self.paths_LQ.append(lq_path)
            else:
                lq_path, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
                self.paths_LQ.append(lq_path)
        self.doCrop = opt['doCrop']
        if 'dataroot_PIX' in opt.keys():
            self.paths_PIX, self.sizes_PIX = util.get_image_paths(self.data_type, opt['dataroot_PIX'])
        # dataroot_GAN is an alternative source of LR images specifically for use in computing the GAN loss, where
        # LR and HR do not need to be paired.
        if 'dataroot_GAN' in opt.keys():
            self.paths_GAN, self.sizes_GAN = util.get_image_paths(self.data_type, opt['dataroot_GAN'])
            print('loaded %i images for use in training GAN only.' % (self.sizes_GAN,))

        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ[0]) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ[0]), len(self.paths_GT))
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        if 'dataroot_PIX' in self.opt.keys():
            self.PIX_env = lmdb.open(self.opt['dataroot_PIX'], readonly=True, lock=False, readahead=False,
                                    meminit=False)

    def motion_blur(self, image, size, angle):
        k = np.zeros((size, size), dtype=np.float32)
        k[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
        k = k * (1.0 / np.sum(k))
        return cv2.filter2D(image, -1, k)

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['target_size']

        # get GT image
        GT_path = self.paths_GT[index]
        resolution = [int(s) for s in self.sizes_GT[index].split('_')
                      ] if self.data_type == 'lmdb' else None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        if self.opt['phase'] != 'train':  # modcrop in the validation / test phase
            img_GT = util.modcrop(img_GT, scale)
        if self.opt['color']:  # change color space if necessary
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # get the pix image
        if self.paths_PIX is not None:
            PIX_path = self.paths_PIX[index]
            img_PIX = util.read_img(self.PIX_env, PIX_path, resolution)
            if self.opt['color']:  # change color space if necessary
                img_PIX = util.channel_convert(img_PIX.shape[2], self.opt['color'], [img_PIX])[0]
        else:
            img_PIX = img_GT

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.get_lq_path(index)
            resolution = [int(s) for s in self.sizes_LQ[index].split('_')
                          ] if self.data_type == 'lmdb' else None
            img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(img_GT, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape

            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        img_GAN = None
        if self.paths_GAN:
            GAN_path = self.paths_GAN[index % self.sizes_GAN]
            img_GAN = util.read_img(self.LQ_env, GAN_path)

        # Enforce force_resize constraints.
        h, w, _ = img_LQ.shape
        if h % self.force_multiple != 0 or w % self.force_multiple != 0:
            h, w = (w - w % self.force_multiple), (h - h % self.force_multiple)
            img_LQ = cv2.resize(img_LQ, (h, w))
            h *= scale
            w *= scale
            img_GT = cv2.resize(img_GT, (h, w))
            img_PIX = cv2.resize(img_LQ, (h, w))

        if self.opt['phase'] == 'train':
            H, W, _ = img_GT.shape
            assert H >= GT_size and W >= GT_size

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale

            if self.doCrop:
                # randomly crop
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                if img_GAN is not None:
                    img_GAN = img_GAN[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
                img_PIX = img_PIX[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            else:
                img_LQ = cv2.resize(img_LQ, (LQ_size, LQ_size), interpolation=cv2.INTER_LINEAR)
                if img_GAN is not None:
                    img_GAN = cv2.resize(img_GAN, (LQ_size, LQ_size), interpolation=cv2.INTER_LINEAR)
                img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
                img_PIX = cv2.resize(img_PIX, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)

                if 'doResizeLoss' in self.opt.keys() and self.opt['doResizeLoss']:
                    r = random.randrange(0, 10)
                    if r > 5:
                        img_LQ = cv2.resize(img_LQ, (int(LQ_size/2), int(LQ_size/2)), interpolation=cv2.INTER_LINEAR)
                        img_LQ = cv2.resize(img_LQ, (LQ_size, LQ_size), interpolation=cv2.INTER_LINEAR)

            # augmentation - flip, rotate
            img_LQ, img_GT, img_PIX = util.augment([img_LQ, img_GT, img_PIX], self.opt['use_flip'],
                                          self.opt['use_rot'])

            if self.opt['use_blurring']:
                # Pick randomly between gaussian, motion, or no blur.
                blur_det = random.randint(0, 100)
                blur_magnitude = 3 if 'blur_magnitude' not in self.opt.keys() else self.opt['blur_magnitude']
                if blur_det < 40:
                    blur_sig = int(random.randrange(0, blur_magnitude))
                    img_LQ = cv2.GaussianBlur(img_LQ, (blur_magnitude, blur_magnitude), blur_sig)
                elif blur_det < 70:
                    img_LQ = self.motion_blur(img_LQ, random.randrange(1, blur_magnitude * 3), random.randint(0, 360))


        if self.opt['color']:  # change color space if necessary
            img_LQ = util.channel_convert(C, self.opt['color'],
                                          [img_LQ])[0]  # TODO during val no definition

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = cv2.cvtColor(img_GT, cv2.COLOR_BGR2RGB)
            img_LQ = cv2.cvtColor(img_LQ, cv2.COLOR_BGR2RGB)
            if img_GAN is not None:
                img_GAN = cv2.cvtColor(img_GAN, cv2.COLOR_BGR2RGB)
            img_PIX = cv2.cvtColor(img_PIX, cv2.COLOR_BGR2RGB)

        # LQ needs to go to a PIL image to perform the compression-artifact transformation.
        img_LQ = (img_LQ * 255).astype(np.uint8)
        img_LQ = Image.fromarray(img_LQ)
        if self.opt['use_compression_artifacts'] and random.random() > .25:
            qf = random.randrange(10, 70)
            corruption_buffer = BytesIO()
            img_LQ.save(corruption_buffer, "JPEG", quality=qf, optimice=True)
            corruption_buffer.seek(0)
            img_LQ = Image.open(corruption_buffer)

        if 'grayscale' in self.opt.keys() and self.opt['grayscale']:
            img_LQ = ImageOps.grayscale(img_LQ).convert('RGB')

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_PIX = torch.from_numpy(np.ascontiguousarray(np.transpose(img_PIX, (2, 0, 1)))).float()
        img_LQ = F.to_tensor(img_LQ)
        if img_GAN is not None:
            img_GAN = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GAN, (2, 0, 1)))).float()

        lq_noise = torch.randn_like(img_LQ) * 5 / 255
        img_LQ += lq_noise

        if LQ_path is None:
            LQ_path = GT_path
        d = {'LQ': img_LQ, 'GT': img_GT, 'PIX': img_PIX, 'LQ_path': LQ_path, 'GT_path': GT_path}
        if img_GAN is not None:
            d['GAN'] = img_GAN
        return d

    def __len__(self):
        return len(self.paths_GT)
