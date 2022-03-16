import PIL.Image
import zipfile
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


class ZipFileDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.path = opt['path']
        zip = zipfile.ZipFile(self.path)
        self.all_files = list(zip.namelist())
        self.resolution = opt['resolution']
        self.paired_mode = opt['paired_mode']
        self.transforms = Compose([ToTensor(),
                                 Resize(self.resolution),
                                 Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                  ])
        self.zip = None

    def __len__(self):
        return len(self.all_files)

    # Loaded on the fly because ZipFile does not tolerate pickling.
    def get_zip(self):
        if self.zip is None:
            self.zip = zipfile.ZipFile(self.path)
        return self.zip

    def load_image(self, path):
        file = self.get_zip().open(path, 'r')
        pilimg = PIL.Image.open(file)
        tensor = self.transforms(pilimg)
        return tensor

    def __getitem__(self, i):
        try:
            fname = self.all_files[i]
            out = {
                'hq': self.load_image(fname),
                'HQ_path': fname,
                'has_alt': self.paired_mode
            }
            if self.paired_mode:
                if fname.endswith('0.jpg'):
                    aname = fname.replace('0.jpg', '1.jpg')
                else:
                    aname = fname.replace('1.jpg', '0.jpg')
                out['alt_hq'] = self.load_image(aname)
        except:
            print(f"Error loading {fname} from zipfile. Attempting to recover by loading next element.")
            return self[i+1]
        return out

if __name__ == '__main__':
    opt = {
        'path': 'E:\\4k6k\\datasets\\images\\youtube-imagenet-paired\\output.zip',
        'resolution': 224,
        'paired_mode': True
    }
    dataset = ZipFileDataset(opt)
    print(len(dataset))
    loader = DataLoader(dataset, shuffle=True)
    for i, d in enumerate(loader):
        torchvision.utils.save_image(d['hq'], f'{i}_hq.png')
        torchvision.utils.save_image(d['alt_hq'], f'{i}_althq.png')

