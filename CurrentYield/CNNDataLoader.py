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

class DataSplitter():
    
    def __call__(self, csv_path, split=[0.8, 0.2]):
        
        data = pd.read_csv(csv_path)
                    
        # pylint: disable=E1101
        mask = np.random.rand(len(data)) < split[0]
        # pylint: enable=E1101

        train = data[mask]
        test  = data[~mask]

        return train, test


class CNNDataSet(Dataset):
    
    def __init__(self, dataframe, image_dir, transform=None):
        
        self.image_dir = image_dir
        self.transforms  = transform
        self.annotations = dataframe

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        annotation   = self.annotations.iloc[index,:]

        target = annotation.loc['dry wt (mg)'].astype('float')

        # Get the image out
        tray     = str(int(annotation.loc['tray']))
        position = str(annotation.loc['position'])
        name     = str(annotation.loc['name'])

        path = os.path.join(self.image_dir, 'PSI_Tray0' + tray, position, name)
        image = io.imread(path)

        datapoint = {'target' : target, 'image' : image}
        
        if self.transforms:
            datapoint = self.transforms(datapoint)

        return datapoint


class SquareRescale(object):
    
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        
        
        image = transform.resize(image, (int(self.size), int(self.size)))

        return {'image': image, 'target' : target}

class ToTensor(object):
    """ Convert Numpy array to pytorch image array"""
    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        
        # Swap the color index to suit pytorch
        image = image.transpose((2,0,1))

        # pylint: disable=E1101
        return {'image' : torch.from_numpy(image),
                'target' :target}
        # pylint: enable=E1101



class GuessTransform(object):
    
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = transform.resize(image, (int(self.size), int(self.size)))
        image = image.transpose((2,0,1))
        # pylint: disable=E1101
        return torch.from_numpy(image)
        # pylint: enable=E1101



if __name__ == '__main__':

    ds = DataSplitter()
    train, test = ds('../../data/harvested.csv')

    train_dataset = CNNDataSet(train, '../../../images_and_annotations', transform=None)

                                    
                                        

