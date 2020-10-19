import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import os

import utils
import utils.options as option
import utils.util as util
from data.util import bgr2ycbcr
import models.archs.SwitchedResidualGenerator_arch as srg
from models.ExtensibleTrainer import ExtensibleTrainer
from switched_conv.switched_conv_util import save_attention_to_image, save_attention_to_image_rgb
from switched_conv.switched_conv import compute_attention_specificity
from data import create_dataset, create_dataloader
from tqdm import tqdm
import torch
import models.networks as networks


# Concepts: Swap transformations around. Normalize attention. Disable individual switches, both randomly and one at
# a time, starting at the last switch. Pick random regions in an image and print out the full attention vector for
# each switch. Yield an output directory name for each alteration and None when last alteration is completed.
def alter_srg(srg: srg.ConfigurableSwitchedResidualGenerator2):
    # First alteration, strip off switches one at a time.
    yield "naked"

    '''
    for i in range(1, len(srg.switches)):
        srg.switches = srg.switches[:-i]
        yield "stripped-%i" % (i,)
    '''

    for sw in srg.switches:
        sw.set_temperature(.001)
    yield "specific"

    for sw in srg.switches:
        sw.set_temperature(1000)
    yield "normalized"

    for sw in srg.switches:
        sw.set_temperature(1)
        sw.switch.attention_norm = None
    yield "no_anorm"
    return None

def analyze_srg(srg: srg.ConfigurableSwitchedResidualGenerator2, path, alteration_suffix):
    mean_hists = [compute_attention_specificity(att, 2) for att in srg.attentions]
    means = [i[0] for i in mean_hists]
    hists = [torch.histc(i[1].clone().detach().cpu().flatten().float(), bins=srg.transformation_counts) for i in mean_hists]
    hists = [h / torch.sum(h) for h in hists]
    for i in range(len(means)):
        print("%s - switch_%i_specificity" % (alteration_suffix, i), means[i])
        print("%s - switch_%i_histogram" % (alteration_suffix, i), hists[i])

    [save_attention_to_image_rgb(path, srg.attentions[i], srg.transformation_counts, alteration_suffix, i) for i in range(len(srg.attentions))]


def forward_pass(model, output_dir, alteration_suffix=''):
    model.feed_data(data, need_GT=need_GT)
    model.test()

    visuals = model.get_current_visuals(need_GT)['rlt'].cpu()
    fea_loss = 0
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

        if need_GT:
            fea_loss += model.compute_fea_loss(visuals[i], data['GT'][i])

        util.save_img(sr_img, save_img_path)
    return fea_loss


if __name__ == "__main__":
    #### options
    torch.backends.cudnn.benchmark = True
    srg_analyze = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../options/srgan_compute_feature.yml')
    opt = option.parse(parser.parse_args().opt, is_train=False)
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
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = ExtensibleTrainer(opt)
    fea_loss = 0
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
                    alt_path = osp.join(dataset_dir, alteration_suffix)
                    img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
                    img_name = osp.splitext(osp.basename(img_path))[0] + opt['name']
                    alteration_suffix += img_name
                    os.makedirs(alt_path, exist_ok=True)
                    forward_pass(model, dataset_dir, alteration_suffix)
                    analyze_srg(model_copy, alt_path, alteration_suffix)
                # Reset model and do next alteration.
                model_copy = networks.define_G(opt).to(model.device)
                model_copy.load_state_dict(orig_model.state_dict())
                model.netG = model_copy
            else:
                fea_loss += forward_pass(model, dataset_dir, opt['name'])

        # log
        logger.info('# Validation # Fea: {:.4e}'.format(fea_loss / len(test_loader)))
