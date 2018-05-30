
import os
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import mean_absolute_error
from skimage import io, transform

import pandas as pd
import numpy as np


class PlantWeightModel(object):
    
    def __init__(self, optimizer_params=None, model_name='cnn.pkl', image_dir='./../images_and_annotations/', 
            data_path= './data/harvested.csv', 
            batch_size=1, num_epochs=20, image_size=200, 
            load_model=False, save_model=True, gpu_en=True):
        
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.image_size = image_size
        
        if  optimizer_params is None:
            self.optimizer_params  = {'learning_rate' : 0.015,
                                      'momentum' : 0.9,
                                      'decay' : 2,
                                      'amsgrad' : False}        
        else:
            self.optimizer_params = optimizer_params 

        # data paths
        self.image_dir = image_dir
        self.data_path = data_path
        self.init_dataset()


        # Model init
        self.model_name = model_name
        self.gpu_en = gpu_en and torch.cuda.is_available()
        self.model =  PlantWieghtRegressor()
        self.save_model = save_model # Choose whether or not training the model saves the data
        if self.gpu_en:
            self.model = self.model.cuda()

        self.init = False

        if load_model:
            self.init = True
            self.model.load_state_dict(torch.load('cnn.pkl'))
            self.model = self.model.cuda()

    def init_dataset(self):
        
        splitter = DataSplitter()
        train, test  = splitter(self.data_path)

        self.train_dataset = CNNDataSet( train,
                                    image_dir=self.image_dir,
                                    transform=transforms.Compose([SquareRescale(self.image_size),
                                                                    ToTensor()]))

        self.test_dataset = CNNDataSet(  test,
                                    image_dir=self.image_dir,
                                    transform=transforms.Compose([SquareRescale(self.image_size),
                                                                    ToTensor()]))
        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                batch_size=self.batch_size, 
                                                shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                batch_size=self.batch_size, 
                                                shuffle=False)

    def Train(self):
        print('Training New Model...')
        # Loss and Optimizer
        criterion = nn.MSELoss(size_average=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_params['learning_rate'],
                    weight_decay=self.optimizer_params['decay'], amsgrad=self.optimizer_params['amsgrad'])

        mae_history = list()
        mse_history = list()
        for epoch in range(self.num_epochs):
            for i, data in enumerate(self.train_loader):
                images = Variable(data['image']).float()
                targets = Variable(data['target']).float()

                if self.gpu_en:
                    images  = images.cuda()
                    targets = targets.cuda()

                output = self.model(images)  
                # print('Output', output)
                # print('Target', target)
                loss = criterion(output, targets) 

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            y_labels        = targets.cpu().numpy() if self.gpu_en else targets.numpy()
            y_pred_results  = output.cpu().data.numpy() if self.gpu_en else output.data.numpy()

            error = mean_absolute_error(y_labels, y_pred_results)
  
            mae_history.append(error)
            mse_history.append(loss.data[0].cpu().numpy()) if self.gpu_en else mse_history.append(loss.data[0].numpy())

            print ('Epoch: {}, MA Error {:.2f}, MSE Loss {:.2f}'.format(epoch, error, loss.data[0]))
        
        self.model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).

        if self.save_model:
            torch.save(self.model.state_dict(), self.model_name)

        self.init = True

        return {'num_epochs': self.num_epochs, 
                'mae_history': mae_history, 
                'mse_history' : mse_history}
        


    def Test(self):
        print('Testing new Model')

        mean_error = 0
        total_error = 0
        num_batches = 0
        history = list()
        for data in self.test_loader:
            images = Variable(data['image']).float()
            targets = Variable(data['target']).float()

            if self.gpu_en:
                images = images.cuda()

            outputs = self.model(images)

            y_labels = targets.cpu().numpy() if self.gpu_en else targets.numpy()
            y_pred_results = outputs.cpu().data.numpy() if self.gpu_en else y_labels.data.numpy()

            mean_error = mean_absolute_error(y_labels, y_pred_results)
            print(y_labels)
            print(y_pred_results)
            num_batches += 1
            total_error += mean_error 

            history.append(mean_error)

        print('Test Accuracy of the model on the 10000 test images: %d %%' % ( total_error / num_batches))
        return {'average_error' : total_error/num_batches, "history":history}

    def Guess(self, raw_image):
        # Assert the init has been made
        if self.init:
            image_loader = GuessTransform(self.image_size)
            image_tensor = Variable(image_loader(raw_image)).float().unsqueeze_(0)

            if self.gpu_en:
                image_tensor = image_tensor.cuda()

            output = self.model(image_tensor)

            return round(output.cpu().data.numpy()[0], -1) if self.gpu_en else round(output.data.numpy()[0], -1)


class PlantWieghtRegressor(nn.Module):
    def __init__(self):
        super(PlantWieghtRegressor, self).__init__()
        self.conv1 = nn.Conv2d(3, 5,  kernel_size=2)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=2)
        self.mp = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(24010, 5000)
        self.fc2 = nn.Linear(5000 , 500)
        self.fc3 = nn.Linear(500  , 100)
        self.fc4 = nn.Linear(100  , 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        in_size = x.size(0)
        batch_size = len(x)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.mp(F.relu(self.conv2(x)))
        x = x.view(in_size, -1)

        fc = [self.fc1, self.fc2, self.fc3, self.fc4]

        for i in range(len(fc)):
            layer = fc[i]
            if i < len(fc) - 1:
                x = self.relu(layer(x))
            else:
                x = layer(x)
        return x.view(batch_size)



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
        return {'image' :  torch.from_numpy(image),
                'target' : target}
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
    
    load_model = False

    WeightPredictor = PlantWeightModel(load_model=load_model, save_model=False)
    if not load_model:
        WeightPredictor.Train()
        WeightPredictor.Test()

    image_path = '../../images_and_annotations/PSI_Tray031/p-1/PSI_Tray031_2016-01-17--13-28-52_top_1-1_930.png'
    image = io.imread(image_path)
    
    print("Estimated Weight: {}".format(WeightPredictor.Guess(image)))



    # def __init__(self, model_name='cnn.pkl', image_dir='../../../images_and_annotations/', 
    #             data_path='../../../data/harvested.csv',
    #             batch_size=10, num_epochs=10, learning_rate=0.001, image_size=200,
    #             load_model=True, save_model=True, gpu_en=True):