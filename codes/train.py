import os
import math
import argparse
import random
import logging
import shutil
from tqdm import tqdm

import torch
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from time import time


def init_dist(backend='nccl', **kwargs):
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

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.', default='../options/train_imgset_spsr3_gan.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    colab_mode = False if 'colab_mode' not in opt.keys() else opt['colab_mode']
    if colab_mode:
        # Check the configuration of the remote server. Expect models, resume_state, and val_images directories to be there.
        # Each one should have a TEST file in it.
        util.get_files_from_server(opt['ssh_server'], opt['ssh_username'], opt['ssh_password'],
                                   os.path.join(opt['remote_path'], 'training_state', "TEST"))
        util.get_files_from_server(opt['ssh_server'], opt['ssh_username'], opt['ssh_password'],
                                   os.path.join(opt['remote_path'], 'models', "TEST"))
        util.get_files_from_server(opt['ssh_server'], opt['ssh_username'], opt['ssh_password'],
                                   os.path.join(opt['remote_path'], 'val_images', "TEST"))
        # Load the state and models needed from the remote server.
        if opt['path']['resume_state']:
            util.get_files_from_server(opt['ssh_server'], opt['ssh_username'], opt['ssh_password'], os.path.join(opt['remote_path'], 'training_state', opt['path']['resume_state']))
        if opt['path']['pretrain_model_G']:
            util.get_files_from_server(opt['ssh_server'], opt['ssh_username'], opt['ssh_password'], os.path.join(opt['remote_path'], 'models', opt['path']['pretrain_model_G']))
        if opt['path']['pretrain_model_D']:
            util.get_files_from_server(opt['ssh_server'], opt['ssh_username'], opt['ssh_password'], os.path.join(opt['remote_path'], 'models', opt['path']['pretrain_model_D']))

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

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
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            tb_logger_path = os.path.join(opt['path']['experiments_root'], 'tb_logger')
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir=tb_logger_path)
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 1  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = -1 if 'start_step' not in opt.keys() else opt['start_step']
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        tq_ldr = tqdm(train_loader)

        _t = time()
        _profile = False
        for _, train_data in enumerate(tq_ldr):
            if _profile:
                print("Data fetch: %f" % (time() - _t))
                _t = time()

            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            if _profile:
                print("Update LR: %f" % (time() - _t))
                _t = time()
            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            if _profile:
                print("Model feed + step: %f" % (time() - _t))
                _t = time()

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log(current_step)
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    if 'histogram' in k:
                        if rank <= 0:
                            tb_logger.add_histogram(k, v, current_step)
                    else:
                        message += '{:s}: {:.4e} '.format(k, v)
                        # tensorboard logger
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            if rank <= 0:
                                tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)
            #### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                if opt['model'] in ['sr', 'srgan', 'corruptgan', 'spsrgan', 'extensibletrainer'] and rank <= 0:  # image restoration validation
                    model.force_restore_swapout()
                    val_batch_sz = 1 if 'batch_size' not in opt['datasets']['val'].keys() else opt['datasets']['val']['batch_size']
                    # does not support multi-GPU validation
                    pbar = util.ProgressBar(len(val_loader) * val_batch_sz)
                    avg_psnr = 0.
                    avg_fea_loss = 0.
                    idx = 0
                    colab_imgs_to_copy = []
                    for val_data in val_loader:
                        idx += 1
                        for b in range(len(val_data['LQ_path'])):
                            img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][b]))[0]
                            img_dir = os.path.join(opt['path']['val_images'], img_name)
                            util.mkdir(img_dir)

                            model.feed_data(val_data)
                            model.test()

                            visuals = model.get_current_visuals()
                            if visuals is None:
                                continue

                            sr_img = util.tensor2img(visuals['rlt'][b])  # uint8
                            #gt_img = util.tensor2img(visuals['GT'][b])  # uint8

                            # Save SR images for reference
                            img_base_name = '{:s}_{:d}.png'.format(img_name, current_step)
                            save_img_path = os.path.join(img_dir, img_base_name)
                            util.save_img(sr_img, save_img_path)
                            if colab_mode:
                                colab_imgs_to_copy.append(save_img_path)

                            # calculate PSNR (Naw - don't do that. PSNR sucks)
                            #sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
                            #avg_psnr += util.calculate_psnr(sr_img, gt_img)
                            #pbar.update('Test {}'.format(img_name))

                            # calculate fea loss
                            avg_fea_loss += model.compute_fea_loss(visuals['rlt'][b], visuals['GT'][b])

                    if colab_mode:
                        util.copy_files_to_server(opt['ssh_server'], opt['ssh_username'], opt['ssh_password'],
                                                  colab_imgs_to_copy,
                                                  os.path.join(opt['remote_path'], 'val_images', img_base_name))

                    avg_psnr = avg_psnr / idx
                    avg_fea_loss = avg_fea_loss / idx

                    # log
                    logger.info('# Validation # PSNR: {:.4e} Fea: {:.4e}'.format(avg_psnr, avg_fea_loss))
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        #tb_logger.add_scalar('val_psnr', avg_psnr, current_step)
                        tb_logger.add_scalar('val_fea', avg_fea_loss, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)
                if 'alt_path' in opt['path'].keys():
                    import shutil
                    print("Synchronizing tb_logger to alt_path..")
                    alt_tblogger = os.path.join(opt['path']['alt_path'], "tb_logger")
                    shutil.rmtree(alt_tblogger, ignore_errors=True)
                    shutil.copytree(tb_logger_path, alt_tblogger)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        tb_logger.close()


if __name__ == '__main__':
    main()
