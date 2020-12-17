import os
from collections import OrderedDict

import orjson as json
# Given a JSON file produced by the VS.net image labeler utility, produces a dict where the keys are image file names
# and the values are a list of object with the following properties:
# [patch_top, patch_left, patch_height, patch_width, label]
import torch


class VsNetImageLabeler:
    def __init__(self, label_file):
        if not isinstance(label_file, list):
            label_file = [label_file]
        self.labeled_images = {}
        for lfil in label_file:
            with open(lfil, "r") as read_file:
                self.label_file = label_file
                # Format of JSON file:
                # "key_binding" {
                #    "label": "<label>"
                #    "index": <num>
                #    "keyBinding": "key_binding"
                #    "labeledImages": [
                #        { "path", "label", "patch_top", "patch_left", "patch_height", "patch_width" }
                #    ]
                # }
                categories = json.loads(read_file.read())
                available_labels = {}
                label_value_dict = {}
                for cat in categories.values():
                    available_labels[cat['index']] = cat['label']
                    label_value_dict[cat['label']] = cat['index']
                    for lbli in cat['labeledImages']:
                        pth = lbli['path']
                        if pth not in self.labeled_images.keys():
                            self.labeled_images[pth] = []
                        self.labeled_images[pth].append(lbli)

                # Insert "labelValue" for each entry.
                for v in self.labeled_images.values():
                    for l in v:
                        l['labelValue'] = label_value_dict[l['label']]

        self.categories = categories
        self.str_labels = available_labels

    def get_labeled_paths(self, base_path):
        return [os.path.join(base_path, pth) for pth in self.labeled_images]

    def get_labels_as_tensor(self, hq, img_key, resize_factor):
        _, h, w = hq.shape
        labels = torch.zeros((1,h,w), dtype=torch.long)
        mask = torch.zeros((1,h,w), dtype=torch.float)
        lbl_list = self.labeled_images[img_key]
        for patch_lbl in lbl_list:
            t, l, h, w = patch_lbl['patch_top'] // resize_factor, patch_lbl['patch_left'] // resize_factor, \
                         patch_lbl['patch_height'] // resize_factor, patch_lbl['patch_width'] // resize_factor
            val = patch_lbl['labelValue']
            labels[:,t:t+h,l:l+w] = val
            mask[:,t:t+h,l:l+w] = 1.0
        return labels, mask, self.str_labels

    def add_label(self, binding, img_name, top, left, dim):
        lbl = {"path": img_name, "label": self.categories[binding]['label'], "patch_top": top, "patch_left": left,
               "patch_height": dim, "patch_width": dim}
        self.categories[binding]['labeledImages'].append(lbl)

    def save(self):
        with open(self.label_file[0], "wb") as file:
            file.write(json.dumps(self.categories))
