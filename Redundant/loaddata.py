
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

def load_data(base_path):
    
    if not os.path.isdir(base_path):
        print ("Path: ", base_path, " is not a directory")
    
    DataSet = list()

    for i in range(1,20):
        label = 'p-' + str(i)
        filename = 'PSI_Tray031' + label + '.csv'
        path = base_path + '/' + label + '/'
        with open(path + filename) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                print(row)
                DataSet.append(ImageData(path, filename, row))


# def load_data(base_path):

#     planters = list()

#     for i in range(1,20):
#         label = 'p-' + str(i)
#         filename = 'PSI_Tray031' + label + '.csv'
#         path = base_path + '/' + label + '/'
#         planters.append(pd.read_csv(os.path.join(path + filename)))
        
#     print (planters[5])

class ImageData(object):
    
    def __init__(self, path, name, data):
        self.path = path
        self.name = name
        self.id =  data[0]
        self.make_datetime(data[1], data[2])
        self.green = data[3]
    
    def make_datetime(self, date, time):
       
        # Get date in format: yyyy-mm-dd:hh-mm-ss
        date_str = date + ':' + time
        self.datetime = dt.datetime.strptime(date_str, '%Y-%m-%d:%H-%M-%S')

class PlantImageDataset(Dataset):

    def __init__(self, csv_files, image_dir):
        
        self.annotations = pd.DataFrame()
        for file in csv_files:
            self.annotations.append = pd.read_csv(file)

        self.image_dir = image_dir
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_name)
        annotation = self.annotations.iloc[index,2:].as_matrix()
        annotation = annotation.astype('float')
        sample = {'image': image, 'annotation' : annotation}

        return sample


        
if __name__ == "__main__": 
    load_data('../images_and_annotations/PSI_Tray031/')