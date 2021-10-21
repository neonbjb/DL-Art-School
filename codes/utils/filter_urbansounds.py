import os
import shutil

from scipy.io.wavfile import read
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    apath = 'E:\\audio\\UrbanSound\\UrbanSound8K\\audio\\'
    csv_file = open('E:\\audio\\UrbanSound\\UrbanSound8K\\metadata\\UrbanSound8K.csv', 'r')
    csv = csv_file.read()
    csv_file.close()
    for it, line in tqdm(enumerate(csv.splitlines(keepends=False))):
        if it == 0:
            continue
        l = line.split(',')
        f = os.path.join(apath, f'fold{l[5]}', l[0])
        c = l[7]
        try:
            if c in ['children_playing', 'street_music', 'gun_shot']:
                continue
            sampling_rate, data = read(f)
            if data.dtype == np.int32:
                norm_fix = 2 ** 31
            elif data.dtype == np.int16:
                norm_fix = 2 ** 15
            elif data.dtype == np.float16 or data.dtype == np.float32:
                norm_fix = 1.
            else:
                raise NotImplemented(f"Provided data dtype not supported: {data.dtype}")
            shutil.copy(f, os.path.join('E:\\audio\\UrbanSound\\filtered', l[0]))
        except:
            pass