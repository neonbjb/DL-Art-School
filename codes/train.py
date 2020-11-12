import os
import math
import argparse
import random
import logging
from tqdm import tqdm

import torch
from data.data_sampler import DistIterSampler

from utils import util, options as option
from data import create_dataloader, create_dataset
from models.ExtensibleTrainer import ExtensibleTrainer
from time import time

def init_dist(backend, **kwargs):
    # These packages have globals that screw with Windows, so only import them if needed.
    import torch.distributed as dist
    import torch.multiprocessing as mp

    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

class Trainer:

    def init(self, opt, launcher, all_networks={}):
        self._profile = False
        self.val_compute_psnr = opt['eval']['compute_psnr'] if 'compute_psnr' in opt['eval'] else True
        self.val_compute_fea = opt['eval']['compute_fea'] if 'compute_fea' in opt['eval'] else True

        #### wandb init
        if opt['wandb']:
            import wandb
            wandb.init(project=opt['name'])

        #### loading resume state if exists
        if opt['path'].get('resume_state', None):
            # distributed resuming: all load into default GPU
            device_id = torch.cuda.current_device()
            resume_state = torch.load(opt['path']['resume_state'],
                                      map_location=lambda storage, loc: storage.cuda(device_id))
            option.check_resume(opt, resume_state['iter'])  # check resume options
        else:
            resume_state = None

        #### mkdir and loggers
        if self.rank <= 0:  # normal training (self.rank -1) OR distributed training (self.rank 0)
            if resume_state is None:
                util.mkdir_and_rename(
                    opt['path']['experiments_root'])  # rename experiment folder if exists
                util.mkdirs(
                    (path for key, path in opt['path'].items() if not key == 'experiments_root' and path is not None
                     and 'pretrain_model' not in key and 'resume' not in key))

            # config loggers. Before it, the log will not work
            util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                              screen=True, tofile=True)
            self.logger = logging.getLogger('base')
            self.logger.info(option.dict2str(opt))
            # tensorboard logger
            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                self.tb_logger_path = os.path.join(opt['path']['experiments_root'], 'tb_logger')
                version = float(torch.__version__[0:3])
                if version >= 1.1:  # PyTorch 1.1
                    from torch.utils.tensorboard import SummaryWriter
                else:
                    self.self.logger.info(
                        'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                    from tensorboardX import SummaryWriter
                self.tb_logger = SummaryWriter(log_dir=self.tb_logger_path)
        else:
            util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
            self.logger = logging.getLogger('base')

        # convert to NoneDict, which returns None for missing keys
        opt = option.dict_to_nonedict(opt)
        self.opt = opt

        #### random seed
        seed = opt['train']['manual_seed']
        if seed is None:
            seed = random.randint(1, 10000)
        if self.rank <= 0:
            self.logger.info('Random seed: {}'.format(seed))
        util.set_random_seed(seed)

        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True
        # torch.autograd.set_detect_anomaly(True)

        # Save the compiled opt dict to the global loaded_options variable.
        util.loaded_options = opt

        #### create train and val dataloader
        dataset_ratio = 1  # enlarge the size of each epoch
        for phase, dataset_opt in opt['datasets'].items():
            if phase == 'train':
                self.train_set = create_dataset(dataset_opt)
                train_size = int(math.ceil(len(self.train_set) / dataset_opt['batch_size']))
                total_iters = int(opt['train']['niter'])
                self.total_epochs = int(math.ceil(total_iters / train_size))
                if opt['dist']:
                    self.train_sampler = DistIterSampler(self.train_set, self.world_size, self.rank, dataset_ratio)
                    self.total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
                else:
                    self.train_sampler = None
                self.train_loader = create_dataloader(self.train_set, dataset_opt, opt, self.train_sampler)
                if self.rank <= 0:
                    self.logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                        len(self.train_set), train_size))
                    self.logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                        self.total_epochs, total_iters))
            elif phase == 'val':
                self.val_set = create_dataset(dataset_opt)
                self.val_loader = create_dataloader(self.val_set, dataset_opt, opt, None)
                if self.rank <= 0:
                    self.logger.info('Number of val images in [{:s}]: {:d}'.format(
                        dataset_opt['name'], len(self.val_set)))
            else:
                raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
        assert self.train_loader is not None

        #### create model
        self.model = ExtensibleTrainer(opt, cached_networks=all_networks)

        #### resume training
        if resume_state:
            self.logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                resume_state['epoch'], resume_state['iter']))

            self.start_epoch = resume_state['epoch']
            self.current_step = resume_state['iter']
            self.model.resume_training(resume_state, 'amp_opt_level' in opt.keys())  # handle optimizers and schedulers
        else:
            self.current_step = -1 if 'start_step' not in opt.keys() else opt['start_step']
            self.start_epoch = 0
        if 'force_start_step' in opt.keys():
            self.current_step = opt['force_start_step']

    def do_step(self, train_data):
        if self._profile:
            print("Data fetch: %f" % (time() - _t))
            _t = time()

        opt = self.opt
        self.current_step += 1
        #### update learning rate
        self.model.update_learning_rate(self.current_step, warmup_iter=opt['train']['warmup_iter'])

        #### training
        if self._profile:
            print("Update LR: %f" % (time() - _t))
            _t = time()
        self.model.feed_data(train_data)
        self.model.optimize_parameters(self.current_step)
        if self._profile:
            print("Model feed + step: %f" % (time() - _t))
            _t = time()

        #### log
        if self.current_step % opt['logger']['print_freq'] == 0 and self.rank <= 0:
            logs = self.model.get_current_log(self.current_step)
            message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(self.epoch, self.current_step)
            for v in self.model.get_current_learning_rate():
                message += '{:.3e},'.format(v)
            message += ')] '
            for k, v in logs.items():
                if 'histogram' in k:
                    self.tb_logger.add_histogram(k, v, self.current_step)
                elif isinstance(v, dict):
                    self.tb_logger.add_scalars(k, v, self.current_step)
                else:
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        self.tb_logger.add_scalar(k, v, self.current_step)
            if opt['wandb']:
                import wandb
                wandb.log(logs)
            self.logger.info(message)

        #### save models and training states
        if self.current_step % opt['logger']['save_checkpoint_freq'] == 0:
            if self.rank <= 0:
                self.logger.info('Saving models and training states.')
                self.model.save(self.current_step)
                self.model.save_training_state(self.epoch, self.current_step)
            if 'alt_path' in opt['path'].keys():
                import shutil
                print("Synchronizing tb_logger to alt_path..")
                alt_tblogger = os.path.join(opt['path']['alt_path'], "tb_logger")
                shutil.rmtree(alt_tblogger, ignore_errors=True)
                shutil.copytree(self.tb_logger_path, alt_tblogger)

        #### validation
        if opt['datasets'].get('val', None) and self.current_step % opt['train']['val_freq'] == 0:
            if opt['model'] in ['sr', 'srgan', 'corruptgan', 'spsrgan',
                                'extensibletrainer'] and self.rank <= 0:  # image restoration validation
                avg_psnr = 0.
                avg_fea_loss = 0.
                idx = 0
                val_tqdm = tqdm(self.val_loader)
                for val_data in val_tqdm:
                    idx += 1
                    for b in range(len(val_data['GT_path'])):
                        img_name = os.path.splitext(os.path.basename(val_data['GT_path'][b]))[0]
                        img_dir = os.path.join(opt['path']['val_images'], img_name)
                        util.mkdir(img_dir)

                        self.model.feed_data(val_data)
                        self.model.test()

                        visuals = self.model.get_current_visuals()
                        if visuals is None:
                            continue

                        sr_img = util.tensor2img(visuals['rlt'][b])  # uint8
                        # calculate PSNR
                        if self.val_compute_psnr:
                            gt_img = util.tensor2img(visuals['GT'][b])  # uint8
                            sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
                            avg_psnr += util.calculate_psnr(sr_img, gt_img)

                        # calculate fea loss
                        if self.val_compute_fea:
                            avg_fea_loss += self.model.compute_fea_loss(visuals['rlt'][b], visuals['GT'][b])

                        # Save SR images for reference
                        img_base_name = '{:s}_{:d}.png'.format(img_name, self.current_step)
                        save_img_path = os.path.join(img_dir, img_base_name)
                        util.save_img(sr_img, save_img_path)

                avg_psnr = avg_psnr / idx
                avg_fea_loss = avg_fea_loss / idx

                # log
                self.logger.info('# Validation # PSNR: {:.4e} Fea: {:.4e}'.format(avg_psnr, avg_fea_loss))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name'] and self.rank <= 0:
                    self.tb_logger.add_scalar('val_psnr', avg_psnr, self.current_step)
                    self.tb_logger.add_scalar('val_fea', avg_fea_loss, self.current_step)

    def do_training(self):
        self.logger.info('Start training from epoch: {:d}, iter: {:d}'.format(self.start_epoch, self.current_step))
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self.epoch = epoch
            if opt['dist']:
                self.train_sampler.set_epoch(epoch)
            tq_ldr = tqdm(self.train_loader)

            _t = time()
            for train_data in tq_ldr:
                self.do_step(train_data)

    def create_training_generator(self, index):
        self.logger.info('Start training from epoch: {:d}, iter: {:d}'.format(self.start_epoch, self.current_step))
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self.epoch = epoch
            if self.opt['dist']:
                self.train_sampler.set_epoch(epoch)
            tq_ldr = tqdm(self.train_loader, position=index)

            _t = time()
            for train_data in tq_ldr:
                yield self.model
                self.do_step(train_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.', default='../options/train_adalatent_mi1_rrdb4x_6bl_pyrrrdb_disc.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    trainer = Trainer()

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        trainer.rank = -1
        if len(opt['gpu_ids']) == 1:
            torch.cuda.set_device(opt['gpu_ids'][0])
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist('nccl')
        trainer.world_size = torch.distributed.get_world_size()
        trainer.rank = torch.distributed.get_rank()

    trainer.init(opt, args.launcher)
    trainer.do_training()
