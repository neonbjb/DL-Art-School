"""A multi-thread tool to crop large images to sub-images for faster IO."""
import os
import os.path as osp
import shutil
import subprocess
from time import sleep

import munch
import numpy as np
import cv2
import torchvision
from PIL import Image
import data.util as data_util  # noqa: E402
import torch.utils.data as data
from tqdm import tqdm
import torch
import random

from models.flownet2.networks.resample2d_package.resample2d import Resample2d
from models.optical_flow.PWCNet import pwc_dc_net


def main():
    opt = {}
    opt['n_thread'] = 0
    opt['compression_level'] = 95  # JPEG compression quality rating.
    opt['dest'] = 'file'
    opt['input_folder'] = 'D:\\dlas\\codes\\scripts\\test'
    opt['save_folder'] = 'D:\\dlas\\codes\\scripts\\test_out'
    opt['imgsize'] = 256
    opt['bottom_crop'] = .1
    opt['keep_folder'] = False

    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))

    go(opt)


def is_video(filename):
    return any(filename.endswith(extension) for extension in ['.mp4', '.MP4', '.avi', '.AVI', '.mkv', '.MKV', '.wmv', '.WMV'])


def get_videos_in_path(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    videos = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_video(fname):
                videos.append(os.path.join(dirpath, fname))
    return videos

def get_time_for_secs(secs):
    mins = int(secs / 60)
    hours = int(mins / 60)
    secs = secs - (mins * 60) - (hours * 3600)
    mins = mins % 60
    return '%02d:%02d:%06.3f' % (hours, mins, secs)

class VideoClipDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        input_folder = opt['input_folder']
        self.videos = get_videos_in_path(input_folder)
        print("Found %i videos" % (len(self.videos),))

    def __getitem__(self, index):
        return self.get(index)

    def extract_n_frames(self, video_file, dest, time_seconds, n):
        ffmpeg_args = ['ffmpeg', '-y', '-ss', get_time_for_secs(time_seconds), '-i', video_file, '-vframes', str(n), f'{dest}/%d.jpg']
        process = subprocess.Popen(ffmpeg_args, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        process.wait()

    def get_video_length(self, video_file):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                             "default=noprint_wrappers=1", video_file],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
        return float(result.stdout.decode('utf-8').strip().replace("duration=", ""))

    def get_image_tensor(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # Access exceptions happen regularly, probably due to the subprocess not fully terminating.
        for tries in range(5):
            try:
                os.remove(path)
                break
            except:
                if tries >= 4:
                    assert False
                else:
                    sleep(.1)

        assert img is not None
        assert len(img.shape) > 2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        # Crop off excess so image dimensions are a multiple of 64.
        h, w, _ = img.shape
        img = img[:(h//64)*64,:(w//64)*64,:]
        return torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()

    def get(self, index):
        path = self.videos[index]
        out_folder = self.opt['save_folder']

        vid_len = int(self.get_video_length(path))
        start = 2
        img_runs = []
        while start < vid_len:
            frames_out = os.path.join(out_folder, f'{index}_{start}')
            os.makedirs(frames_out, exist_ok=False)
            n = random.randint(5, 30)
            self.extract_n_frames(path, frames_out, start, n)
            frames = data_util.find_files_of_type('img', frames_out)[0]
            assert len(frames) == n
            img_runs.append(([self.get_image_tensor(frame) for frame in frames], frames_out))
            start += random.randint(2,5)

        return img_runs

    def __len__(self):
        return len(self.videos)


def compute_flow_and_cleanup(flownet, runs):
    resampler = Resample2d().cuda()
    for run in runs:
        run, path = run
        consolidated_flows = None
        a = run[0].unsqueeze(0).cuda()
        img = a
        dbg = a.clone()
        for i in range(1,len(run)):
            img2 = run[i].unsqueeze(0).cuda()
            flow = flownet(torch.cat([img2, img], dim=1))
            flow = torch.nn.functional.interpolate(flow, size=img.shape[2:], mode='bilinear')
            if consolidated_flows is None:
                consolidated_flows = flow
            else:
                consolidated_flows = resampler(flow, -consolidated_flows) + consolidated_flows
            img = img2
            dbg = resampler(dbg, flow)
        torchvision.utils.save_image(dbg, os.path.join(path, "debug.jpg"))
        consolidated_flows = torch.clamp(consolidated_flows / 255, -.5, .5)
        b = run[-1].unsqueeze(0).cuda()
        _, _, h, w = a.shape
        direct_flows = torch.nn.functional.interpolate(torch.clamp(flownet(torch.cat([a, b], dim=1).float()) / 255, -.5, .5), size=img.shape[2:], mode='bilinear')
        # TODO: Reshape image here.
        '''
        # Perform explicit crops first. These are generally used to get rid of watermarks so we dont even want to
        # consider these areas of the image.
        if 'bottom_crop' in self.opt.keys() and self.opt['bottom_crop'] > 0:
            bc = self.opt['bottom_crop']
            if bc > 0 and bc < 1:
                bc = int(bc * img.shape[0])
            img = img[:-bc, :, :]

        h, w, c = img.shape
        assert min(h,w) >= self.opt['imgsize']

        # We must convert the image into a square.
        dim = min(h, w)
        # Crop the image so that only the center is left, since this is often the most salient part of the image.
        img = img[(h - dim) // 2:dim + (h - dim) // 2, (w - dim) // 2:dim + (w - dim) // 2, :]
        img = cv2.resize(img, (self.opt['imgsize'], self.opt['imgsize']), interpolation=cv2.INTER_AREA)
        '''

        torchvision.utils.save_image(a, os.path.join(path, "a.jpg"))
        torchvision.utils.save_image(b, os.path.join(path, "b.jpg"))
        torch.save(consolidated_flows * 255, os.path.join(path, "consolidated_flow.pt"))
        torchvision.utils.save_image(torch.cat([consolidated_flows + .5, torch.zeros((1, 1, h, w), device='cuda')], dim=1), os.path.join(path, "consolidated_flow.png"))

        # For debugging
        torchvision.utils.save_image(resampler(a, consolidated_flows * 255), os.path.join(path, "b_flowed.jpg"))
        torchvision.utils.save_image(resampler(b, -consolidated_flows * 255), os.path.join(path, "a_flowed.jpg"))
        torchvision.utils.save_image(resampler(b, direct_flows * 255), os.path.join(path, "a_flowed_nonconsolidated.jpg"))
        torchvision.utils.save_image(torch.cat([direct_flows + .5, torch.zeros((1, 1, h, w), device='cuda')], dim=1), os.path.join(path, "direct_flow.png"))


def identity(x):
    return x


def go(opt):
    flownet = pwc_dc_net('../experiments/pwc_humanflow.pth')
    flownet.eval()
    flownet = flownet.cuda()

    dataset = VideoClipDataset(opt)
    dataloader = data.DataLoader(dataset, num_workers=opt['n_thread'], collate_fn=identity)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            compute_flow_and_cleanup(flownet, batch[0])


if __name__ == '__main__':
    main()
