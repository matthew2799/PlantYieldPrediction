
import csv, os, sys
import datetime as dt
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd



class WeightImageDataSet(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):
        
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        img_name = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_name)
        annotation = self.annotations.iloc[index,2:].as_matrix()
        annotation = annotation.astype('float')
        sample = {'image': image, 'annotation' : annotation}

        if self.transform:
            sample = self.transform(sample)

        return sample


