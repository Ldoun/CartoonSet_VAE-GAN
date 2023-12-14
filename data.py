import torch
from torch.utils.data import  Dataset
from torchvision.transforms.functional import to_tensor

import os
from PIL import Image
from glob import glob

class DataSet(Dataset):
    def __init__(self, base_path, label=None):
        self.files = glob(os.path.join(base_path, '*', '*.png'))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        image = to_tensor(Image.open(self.files[index]))
        return image