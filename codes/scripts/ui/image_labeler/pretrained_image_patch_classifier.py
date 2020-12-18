import logging
import os.path as osp

import utils
import utils.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from trainer.ExtensibleTrainer import ExtensibleTrainer


class PretrainedImagePatchClassifier:
    def __init__(self, cfg):
        self.cfg = cfg

        opt = option.parse(cfg, is_train=False)
        opt = option.dict_to_nonedict(opt)
        utils.util.loaded_options = opt

        util.mkdirs(
            (path for key, path in opt['path'].items()
             if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
        util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

        #### Create test dataset and dataloader
        dataset_opt = list(opt['datasets'].values())[0]
        # Remove labeling features from the dataset config and wrappers.
        if 'dataset' in dataset_opt.keys():
            if 'labeler' in dataset_opt['dataset'].keys():
                dataset_opt['dataset']['includes_labels'] = False
                del dataset_opt['dataset']['labeler']
            test_set = create_dataset(dataset_opt)
            if hasattr(test_set, 'wrapped_dataset'):
                test_set = test_set.wrapped_dataset
        else:
            test_set = create_dataset(dataset_opt)
        logger.info('Number of test images: {:d}'.format(len(test_set)))
        self.test_loader = create_dataloader(test_set, dataset_opt, opt)
        self.model = ExtensibleTrainer(opt)
        self.gen = self.model.netsG['generator']
        self.dataset_dir = osp.join(opt['path']['results_root'], opt['name'])
        util.mkdir(self.dataset_dir)

    def get_next_sample(self):

        for data in self.test_loader:
            hq = data['hq'].to('cuda')
            res = self.gen(hq)
            yield hq, res, data

