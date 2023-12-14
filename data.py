import torch
from torch.utils.data import  Dataset
import torchvision.transforms as transform

import os
from PIL import Image
from glob import glob

class DataSet(Dataset):
    def __init__(self, base_path, label=None):
        self.files = glob(os.path.join(base_path, '*', '*.png'))
        self.t = transform.Compose([
            transform.ToTensor(),
            transform.CenterCrop([412, 412]),
            transform.Resize([128, 128], antialias=True)
        ])

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        image = self.t(Image.open(self.files[index]).convert('RGB'))
        return image