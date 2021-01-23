import argparse
import logging
import os
import os.path as osp
import subprocess
import time

import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm

from trainer.ExtensibleTrainer import ExtensibleTrainer
from utils import options as option
import utils.util as util
from data import create_dataloader


class FfmpegBackedVideoDataset(data.Dataset):
    '''Pulls frames from a video one at a time using FFMPEG.'''

    def __init__(self, opt, working_dir):
        super(FfmpegBackedVideoDataset, self).__init__()
        self.opt = opt
        self.video = self.opt['video_file']
        self.working_dir = working_dir
        self.frame_rate = self.opt['frame_rate']
        self.start_at = self.opt['start_at_seconds']
        self.end_at = self.opt['end_at_seconds']
        self.force_multiple = self.opt['force_multiple']
        self.frame_count = (self.end_at - self.start_at) * self.frame_rate
        # The number of (original) video frames that will be stored on the filesystem at a time.
        self.max_working_files = 20

        self.data_type = self.opt['data_type']
        self.vertical_splits = self.opt['vertical_splits'] if 'vertical_splits' in opt.keys() else 1

    def get_time_for_it(self, it):
        secs = it / self.frame_rate + self.start_at
        mins = int(secs / 60)
        hours = int(mins / 60)
        secs = secs - (mins * 60) - (hours * 3600)
        mins = mins % 60
        return '%02d:%02d:%06.3f' % (hours, mins, secs)

    def __getitem__(self, index):
        if self.vertical_splits > 0:
            actual_index = int(index / self.vertical_splits)
        else:
            actual_index = index

        # Extract the frame. Command template: `ffmpeg -ss 17:00.0323 -i <video file>.mp4 -vframes 1 destination.png`
        working_file_name = osp.join(self.working_dir, "working_%d.png" % (actual_index % self.max_working_files,))
        vid_time = self.get_time_for_it(actual_index)
        ffmpeg_args = ['ffmpeg', '-y', '-ss', vid_time, '-i', self.video, '-vframes', '1', working_file_name]
        process = subprocess.Popen(ffmpeg_args, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        process.wait()

        # get LQ image
        LQ_path = working_file_name
        img_LQ = Image.open(LQ_path)
        split_index = (index % self.vertical_splits)
        if self.vertical_splits > 0:
            w, h = img_LQ.size
            w_per_split = int(w / self.vertical_splits)
            left = w_per_split * split_index
            img_LQ = F.crop(img_LQ, 0, left, h, w_per_split)
        img_LQ = F.to_tensor(img_LQ)

        mask = torch.ones(1, img_LQ.shape[1], img_LQ.shape[2])
        ref = torch.cat([img_LQ, mask], dim=0)

        if self.force_multiple > 1:
            assert self.vertical_splits <= 1   # This is not compatible with vertical splits for now.
            c, h, w = img_LQ.shape
            h_, w_ = h, w
            height_removed = h % self.force_multiple
            width_removed = w % self.force_multiple
            if height_removed != 0:
                h_ = self.force_multiple * ((h // self.force_multiple) + 1)
            if width_removed != 0:
                w_ = self.force_multiple * ((w // self.force_multiple) + 1)
            lq_template = torch.zeros(c,h_,w_)
            lq_template[:,:h,:w] = img_LQ
            ref_template = torch.zeros(c,h_,w_)
            ref_template[:,:h,:w] = img_LQ
            img_LQ = lq_template
            ref = ref_template

        return {'lq': img_LQ, 'lq_fullsize_ref': ref,
                'lq_center': torch.tensor([img_LQ.shape[1] // 2, img_LQ.shape[2] // 2], dtype=torch.long) }

    def __len__(self):
        return self.frame_count * self.vertical_splits

def merge_images(files, output_path):
    """Merges several image files together across the vertical axis
    """
    images = [Image.open(f) for f in files]
    w, h = images[0].size

    result_width = w * len(images)
    result_height = h

    result = Image.new('RGB', (result_width, result_height))
    for i in range(len(images)):
        result.paste(im=images[i], box=(i * w, 0))
    result.save(output_path)

if __name__ == "__main__":
    #### options
    torch.backends.cudnn.benchmark = True
    want_just_images = True
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../options/use_video_upsample.yml')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    util.mkdirs(
        (path for key, path in opt['path'].items()
         if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    util.loaded_options = opt

    #### Create test dataset and dataloader
    test_loaders = []

    test_set = FfmpegBackedVideoDataset(opt['dataset'], opt['path']['results_root'])
    test_loader = create_dataloader(test_set, opt['dataset'])
    logger.info('Number of test images in [{:s}]: {:d}'.format(opt['dataset']['name'], len(test_set)))
    test_loaders.append(test_loader)

    model = ExtensibleTrainer(opt)
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    frame_counter = 0
    frames_per_vid = opt['frames_per_mini_vid']
    minivid_crf = opt['minivid_crf']
    vid_output = opt['mini_vid_output_folder'] if 'mini_vid_output_folder' in opt.keys() else dataset_dir
    vid_counter = opt['minivid_start_no'] if 'minivid_start_no' in opt.keys() else 0
    img_index = opt['generator_img_index']
    recurrent_mode = opt['recurrent_mode']
    if recurrent_mode:
        assert opt['dataset']['batch_size'] == 1   # Can only do 1 frame at a time in recurrent mode, by definition.
    scale = opt['scale']
    first_frame = True
    ffmpeg_proc = None

    tq = tqdm(test_loader)
    for data in tq:
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True

        if recurrent_mode and first_frame:
            b, c, h, w = data['lq'].shape
            recurrent_entry = torch.zeros((b,c,h*scale,w*scale), device=data['lq'].device)
            # Optionally swap out the 'generator' for the first frame to create a better image that the recurrent generator works off of.
            if 'recurrent_hr_generator' in opt.keys():
                recurrent_gen = model.env['generators']['generator']
                model.env['generators']['generator'] = model.env['generators'][opt['recurrent_hr_generator']]

        first_frame = False
        if recurrent_mode:
            data['recurrent'] = recurrent_entry

        model.feed_data(data, 0, need_GT=need_GT)
        model.test()
        visuals = model.get_current_visuals()['rlt']

        if recurrent_mode:
            recurrent_entry = visuals
        visuals = visuals.cpu().float()
        for i in range(visuals.shape[0]):
            sr_img = util.tensor2img(visuals[i])  # uint8

            # save images
            save_img_path = osp.join(dataset_dir, '%08d.png' % (frame_counter,))
            util.save_img(sr_img, save_img_path)
            frame_counter += 1

            if frame_counter % frames_per_vid == 0:
                if ffmpeg_proc is not None:
                    print("Waiting for last encode..")
                    ffmpeg_proc.wait()
                print("Encoding minivid %d.." % (vid_counter,))
                # Perform stitching.
                num_splits = opt['dataset']['vertical_splits'] if 'vertical_splits' in opt['dataset'].keys() else 1
                if num_splits > 1:
                    procs = []
                    src_imgs_path = osp.join(dataset_dir, "joined")
                    os.makedirs(src_imgs_path, exist_ok=True)
                    for i in range(int(frames_per_vid / num_splits)):
                        to_join = [osp.join(dataset_dir, "%08d.png" % (j,)) for j in range(i * num_splits, i * num_splits + num_splits)]
                        merge_images(to_join, osp.join(src_imgs_path, "%08d.png" % (i,)))
                else:
                    src_imgs_path = dataset_dir

                # Encoding command line:
                # ffmpeg -framerate 30 -i %08d.png -c:v libx265 -crf 12 -preset slow -pix_fmt yuv444p test.mkv
                cmd = ['ffmpeg', '-y', '-framerate', str(opt['dataset']['frame_rate']), '-f', 'image2', '-i', osp.join(src_imgs_path, "%08d.png"),
                       '-c:v', 'libx265', '-crf', str(minivid_crf), '-preset', 'slow', '-pix_fmt', 'yuv444p', osp.join(vid_output, "mini_%06d.mkv" % (vid_counter,))]
                print(ffmpeg_proc)
                ffmpeg_proc = subprocess.Popen(cmd)#, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                vid_counter += 1
                frame_counter = 0
                print("Done.")


            if want_just_images:
                continue