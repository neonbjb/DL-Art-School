import argparse
import os

import torch
import torchvision

import utils.options as option
from scripts.ui.image_labeler.pretrained_image_patch_classifier import PretrainedImagePatchClassifier

if __name__ == "__main__":
    #### options
    torch.backends.cudnn.benchmark = True
    want_metrics = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../options/train_imgset_structural_classifier.yml')

    classifier = PretrainedImagePatchClassifier(parser.parse_args().opt)
    label_to_search_for = 4
    step = 1
    for hq, res in classifier.get_next_sample():
        res = torch.nn.functional.interpolate(res, size=hq.shape[2:], mode="nearest")
        res_lbl = res[:, label_to_search_for, :, :].unsqueeze(1)
        res_lbl_mask = (1.0 * (res_lbl > .5))*.5 + .5
        hq = hq * res_lbl_mask
        torchvision.utils.save_image(hq, os.path.join(classifier.dataset_dir, "%i.png" % (step,)))
        step += 1
