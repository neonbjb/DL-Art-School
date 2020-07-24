import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
import models.archs.SwitchedResidualGenerator_arch as srg
from switched_conv_util import save_attention_to_image
from data import create_dataset, create_dataloader
from models import create_model
from tqdm import tqdm
import torch
import models.networks as networks


# Concepts: Swap transformations around. Normalize attention. Disable individual switches, both randomly and one at
# a time, starting at the last switch. Pick random regions in an image and print out the full attention vector for
# each switch. Yield an output directory name for each alteration and None when last alteration is completed.
def alter_srg(srg: srg.ConfigurableSwitchedResidualGenerator2):
    # First alteration, strip off switches one at a time.
    yield "naked"
    for i in range(1, len(srg.switches)):
        srg.switches = srg.switches[:-i]
        yield "stripped-%i" % (i,)
    return None

def analyze_srg(srg: srg.ConfigurableSwitchedResidualGenerator2, path, alteration_suffix):
    [save_attention_to_image(path, srg.attentions[i], srg.transformation_counts, i, "attention_" + alteration_suffix,
                             l_mult=5) for i in range(len(srg.attentions))]


def forward_pass(model, output_dir, alteration_suffix=''):
    model.feed_data(data, need_GT=need_GT)
    model.test()

    if isinstance(model.fake_GenOut[0], tuple):
        visuals = model.fake_GenOut[0][0].detach().float().cpu()
    else:
        visuals = model.fake_GenOut[0].detach().float().cpu()
    for i in range(visuals.shape[0]):
        img_path = data['GT_path'][i] if need_GT else data['LQ_path'][i]
        img_name = osp.splitext(osp.basename(img_path))[0]

        sr_img = util.tensor2img(visuals[i])  # uint8

        # save images
        suffix = alteration_suffix
        if suffix:
            save_img_path = osp.join(output_dir, img_name + suffix + '.png')
        else:
            save_img_path = osp.join(output_dir, img_name + '.png')

        util.save_img(sr_img, save_img_path)


if __name__ == "__main__":
    #### options
    torch.backends.cudnn.benchmark = True
    want_just_images = True
    srg_analyze = True
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YMAL file.', default='../options/analyze_srg.yml')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    util.mkdirs(
        (path for key, path in opt['path'].items()
         if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    #### Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = create_model(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []

        tq = tqdm(test_loader)
        for data in tq:
            need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True

            if srg_analyze:
                orig_model = model.netG
                model_copy = networks.define_G(opt).to(model.device)
                model_copy.load_state_dict(orig_model.state_dict())
                model.netG = model_copy
                for alteration_suffix in alter_srg(model_copy):
                    img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
                    img_name = osp.splitext(osp.basename(img_path))[0]
                    alteration_suffix += img_name
                    forward_pass(model, dataset_dir, alteration_suffix)
                    analyze_srg(model_copy, dataset_dir, alteration_suffix)
                # Reset model and do next alteration.
                model_copy = networks.define_G(opt).to(model.device)
                model_copy.load_state_dict(orig_model.state_dict())
                model.netG = model_copy
            else:
                forward_pass(model, dataset_dir)
