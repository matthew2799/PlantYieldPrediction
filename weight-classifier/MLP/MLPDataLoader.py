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
    
    def __call__(self, csv_path):
        
        data = pd.read_csv(csv_path)
        
        # pylint: disable=E1101
        mask = np.random.rand(len(data)) < 0.8
        # pylint: enable=E1101

        train = data[mask]
        test = data[~mask]

        return train, test

class MLPDataSet(Dataset):
    
    def __init__(self, dataframe, transform=None):
        
        self.transforms  = transform
        self.targets = dataframe.loc[:,['dry wt (mg)']]
        self.annotations = dataframe.loc[:,['norm_area', 'cont', 'Xm', 'Ym', 'Zm']]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        target = self.targets.iloc[index].as_matrix().astype('float')
        annotation   = self.annotations.iloc[index,:].as_matrix().astype('float')

        datapoint = {'target':target, 'annotation':annotation}

        if self.transforms:
              datapoint = self.transforms(datapoint)

        return datapoint


if __name__ == '__main__':

    ds = DataSplitter()
    train, test = ds('../../data/harvested.csv')

    train_dataset = MLPDataSet(train, transform=None)

                                    
                                        





