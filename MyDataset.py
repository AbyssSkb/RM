from torch.utils.data import Dataset
import torch
import os
import numpy as np
from PIL import Image
import torchvision
import numpy as np


class MyDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_list = [image for image in os.listdir(self.image_folder) if '.png' in image]

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = torchvision.io.read_image(os.path.join(self.image_folder, self.image_list[index]))
        label_name = self.image_list[index].split('.')[0] + '.txt'
        label = np.loadtxt(os.path.join(self.image_folder, label_name), dtype=int)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, label_name
        