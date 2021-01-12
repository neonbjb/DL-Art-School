from glob import glob

import torch
import os
import shutil

if __name__ == '__main__':
    index_map_file = 'F:\\4k6k\\datasets\\images\\imagenet_2017\\imagenet_index_to_train_folder_name_map.pth'
    ground_truth = 'F:\\4k6k\\datasets\\images\\imagenet_2017\\validation_ground_truth.txt'
    val_path = 'F:\\4k6k\\datasets\\images\\imagenet_2017\\val'

    index_map = torch.load(index_map_file)

    for folder in index_map.values():
        os.makedirs(os.path.join(val_path, folder), exist_ok=True)

    gtfile = open(ground_truth, 'r')
    gtids = []
    for line in gtfile:
        gtids.append(int(line.strip()))
    gtfile.close()

    for i, img_file in enumerate(glob(os.path.join(val_path, "*.JPEG"))):
        shutil.move(img_file, os.path.join(val_path, index_map[gtids[i]],
                                           os.path.basename(img_file)))
    print("Done!")
