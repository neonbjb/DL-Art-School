"""create dataset and dataloader"""
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=True)
    else:
        batch_size = dataset_opt['batch_size'] or 1
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                           pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    # datasets for image restoration
    if mode == 'fullimage':
        from data.full_image_dataset import FullImageDataset as D
    elif mode == 'single_image_extensible':
        from data.single_image_dataset import SingleImageDataset as D
    elif mode == 'multi_frame_extensible':
        from data.multi_frame_dataset import MultiFrameDataset as D
    elif mode == 'combined':
        from data.combined_dataset import CombinedDataset as D
    elif mode == 'multiscale':
        from data.multiscale_dataset import MultiScaleDataset as D
    elif mode == 'paired_frame':
        from data.paired_frame_dataset import PairedFrameDataset as D
    elif mode == 'stylegan2':
        from data.stylegan2_dataset import Stylegan2Dataset as D
    elif mode == 'imagefolder':
        from data.image_folder_dataset import ImageFolderDataset as D
    elif mode == 'torch_dataset':
        from data.torch_dataset import TorchDataset as D
    elif mode == 'byol_dataset':
        from data.byol_attachment import ByolDatasetWrapper as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    return dataset
