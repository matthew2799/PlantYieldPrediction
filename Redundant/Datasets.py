
import csv, os, sys
import datetime as dt
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

class PlantImageDataset(Dataset):

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
        annotation = annotation.astype('double')
        sample = {'image': image, 'annotation' : annotation}

        if self.transform:
            sample = self.transform(sample)

        return sample

    # def get_annotations(self, csv_files):
    #     return pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)


    # def get_default_csvfiles(self):
        
    #     csv_files = list()
    #     base = '../images_and_annotations/PSI_Tray031'
    #     for i in range(1,20):
    #         plot = 'p-' + str(i)
    #         file = 'PSI_Tray031' + plot + '.csv'
    #         csv_files.append(os.path.join(base, plot, file))
        
    #     return csv_files


class SquareRescale(object):
    
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']
        image = transform.resize(image, (int(self.size), int(self.size)))

        return {'image': image, 'annotation' : annotation}

class ToTensor(object):

    def __init__(self, div=True):
        self.div = div
    
    """ Convert Numpy array to pytorch image array"""
    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']
        image = image.transpose((2, 0, 1))

        # pylint: disable=E1101
        return {'image' : torch.from_numpy(image),
                'annotation' : torch.from_numpy(annotation)}
        # pylint: enable=E1101

if __name__ == '__main__':

    test_dataset = PlantImageDataset(csv_file='../images_and_annotations/PSI_Tray031/p-2/PSI_Tray031p-2.csv',
                                    image_dir='../images_and_annotations/PSI_Tray031/p-2',
                                    transform=transforms.Compose([SquareRescale(445),
                                                                 ToTensor()]))
    

    sample = test_dataset[1]
    print(1, sample['image'].size(), sample['annotation'])
