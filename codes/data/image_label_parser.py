import os

import orjson as json
# Given a JSON file produced by the VS.net image labeler utility, produces a dict where the keys are image file names
# and the values are a list of object with the following properties:
# [patch_top, patch_left, patch_height, patch_width, label]
import torch


class VsNetImageLabeler:
    def __init__(self, label_file):
        with open(label_file, "r") as read_file:
            # Format of JSON file:
            # "<nonsense>" {
            #    "label": "<label>"
            #    "keyBinding": "<nonsense>"
            #    "labeledImages": [
            #        { "path", "label", "patch_top", "patch_left", "patch_height", "patch_width" }
            #    ]
            # }
            categories = json.loads(read_file.read())
            labeled_images = {}
            available_labels = []
            for cat in categories.values():
                for lbli in cat['labeledImages']:
                    pth = lbli['path']
                    if pth not in labeled_images.keys():
                        labeled_images[pth] = []
                    labeled_images[pth].append(lbli)
                    if lbli['label'] not in available_labels:
                        available_labels.append(lbli['label'])

            # Build the label values, from [1,inf]
            label_value_dict = {}
            for i, l in enumerate(available_labels):
                label_value_dict[l] = i+1

            # Insert "labelValue" for each entry.
            for v in labeled_images.values():
                for l in v:
                    l['labelValue'] = label_value_dict[l['label']]

            self.labeled_images = labeled_images

    def get_labeled_paths(self, base_path):
        return [os.path.join(base_path, pth) for pth in self.labeled_images]

    def get_labels_as_tensor(self, hq, img_key, resize_factor):
        labels = torch.zeros(hq.shape, dtype=torch.long)
        mask = torch.zeros_like(hq)
        lbl_list = self.labeled_images[img_key]
        for patch_lbl in lbl_list:
            t, l, h, w = patch_lbl['patch_top'] // resize_factor, patch_lbl['patch_left'] // resize_factor, \
                         patch_lbl['patch_height'] // resize_factor, patch_lbl['patch_width'] // resize_factor
            val = patch_lbl['labelValue']
            labels[:,t:t+h,l:l+w] = val
            mask[:,t:t+h,l:l+w] = 1.0
        return labels, mask