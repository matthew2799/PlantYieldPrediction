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

import csv, os, sys
import datetime as dt
import pandas as pd
import numpy as np
from skimage import io, transform

class PlantYieldPredictor():

    def __init__(self, net_name='yield.pkl', data_path='./data/TrainingSet.csv', 
            batch_size=200, num_epochs=100, optimizer_params=None, hidden_size=[1000, 500, 100, 10],
            load_model=False, save_model=True, gpu_en=True):
        
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        if  optimizer_params is None:
            self.optimizer_params  = {'learning_rate' : 0.015,
                                      'momentum' : 0.9,
                                      'decay' : 2,
                                      'amsgrad' : False}        
        else:
            self.optimizer_params = optimizer_params 

        # data paths
        self.data_path = data_path
        self.init_dataset()

        # Model Parameters
        input_size = 2
        output_size = 1

        self.hidden_size = hidden_size

        # Model init
        self.gpu_en = gpu_en and torch.cuda.is_available()
        
        self.net_name = net_name
        self.net =  MLPRegressor(input_layer=input_size, hidden_layers=self.hidden_size, output_layer=output_size)
        self.net.construct_layers()
        
        self.save_model = save_model # Choose whether or not training the model saves the data
        if self.gpu_en:
            self.net = self.net.cuda()

        self.init = False

        if load_model:
            self.init = True
            self.net.load_state_dict(torch.load(self.net_name))
            if self.gpu_en:
                self.net = self.net.cuda()

    def init_dataset(self):
        
        splitter = DataSplitter()
        train, test  = splitter(self.data_path)

        self.train_dataset = MLPDataSet(train)
        self.test_dataset  = MLPDataSet(test)

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                batch_size=self.batch_size, 
                                                shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                batch_size=self.batch_size, 
                                                shuffle=False)

    def Train(self):
        print('Training the model ...')
            
        # Loss and Optimizer
        criterion = nn.MSELoss(size_average=False)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.optimizer_params['learning_rate'],
                    weight_decay=self.optimizer_params['decay'], amsgrad=self.optimizer_params['amsgrad'])

        mae_history = list()
        mse_history = list()
        for epoch in range(self.num_epochs):
            for i, data in enumerate(self.train_loader):
                
                labels  = Variable(data['label']).float()
                target = Variable(data['target']).float()
                
                if self.gpu_en:
                    labels  = labels.cuda()
                    target = target.cuda()

                output = self.net.model(labels)    

                # print('Output', output)
                # print('target', target)
                loss = criterion(output, target) 

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            y_targets = target.cpu().numpy() if self.gpu_en else target.numpy()
            y_pred_results = output.cpu().data.numpy() if self.gpu_en else output.data.numpy()

            error = mean_absolute_error(y_targets, y_pred_results);
            mae_history.append(error)
            mse_history.append(loss.data[0].cpu().numpy()) if self.gpu_en else  mse_history.append(loss.data[0].numpy())

            print ('Epoch: {}, MA Error {:.2f}, MSE Loss {:.2f}'.format(epoch, error, loss.data[0]))
        
        self.net.eval()

        if self.save_model:
            print(self.net_name)
            torch.save(self.net.state_dict(), self.net_name)

        self.init = True
            
        return {'num_epochs': self.num_epochs, 
                'mae_history': mae_history, 
                'mse_history' : mse_history}
        

    def Test(self):
        print('Testing new Model')

        mean_error  = 0
        total_error = 0
        num_batches = 0

        history = list()
        for data in self.test_loader:
            labels = Variable(data['label']).float()
            targets = Variable(data['target']).float()

            if self.gpu_en:
                labels = labels.cuda()

            outputs = self.net.model(labels)

            y_labels= targets.cpu().numpy() if self.gpu_en else targets.numpy()
            y_pred_results = outputs.cpu().data.numpy() if self.gpu_en else outputs.data.numpy()

            mean_error = mean_absolute_error(y_labels, y_pred_results)

            history.append(mean_error)

            print(mean_error)
            num_batches += 1
            total_error += mean_error 

        print('Average Error of the model: %d %%' % ( total_error / num_batches))        
        return {'average_error' : total_error/num_batches, "history":history}

            
    def Guess(self, weight, days):

        if self.init:
            guess = np.array([days, weight])
            print(guess)
            # pylint: disable=E1101
            data_tensor = Variable(torch.from_numpy(guess)).float().unsqueeze_(0)
            # pylint: enable=E1101

            if self.gpu_en:
                data_tensor = data_tensor.cuda()

            output = self.net.model(data_tensor)

            return round(output.cpu().data.numpy()[0][0], -1) if self.gpu_en else round(output.data.numpy()[0][0], -1)

class MLPRegressor(nn.Module):
    
    def __init__(self, input_layer=1, output_layer=1, hidden_layers=[20, 100, 100, 20]):
        super(MLPRegressor, self).__init__()
        print(hidden_layers)
        self.model = None
        self.gpu   = True
        self.input_layer  = input_layer
        self.output_layer = output_layer
        self.hidden_layers = hidden_layers 

    def construct_layers(self):
        self.layers = [self.input_layer] + \
        self.hidden_layers + \
        [self.output_layer]

        self.model = torch.nn.Sequential()

        for index, dimensions in enumerate(self.layers):
            if (index < len(self.layers) - 1):
                module = torch.nn.Linear(dimensions, self.layers[index + 1])
                init.xavier_uniform(module.weight)
                self.model.add_module("linear" + str(index), module)

            if (index < len(self.layers) - 2):
                self.model.add_module("relu" + str(index), torch.nn.ReLU())

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
        
        self.transforms = transform
        self.targets = dataframe.loc[:,['target_dry']]
        self.annotations = dataframe.loc[:,['DTH', 'current_yield']]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        target = self.targets.iloc[index].as_matrix().astype('float')
        label = self.annotations.iloc[index, :].as_matrix().astype('float')
        
        datapoint = {'target':target, 'label': label}

        if self.transforms:
              datapoint = self.transforms(datapoint)

        return datapoint




if __name__ == '__main__':
    load_model = False

    YieldRegressor = PlantYieldPredictor(load_model=load_model)
    if not load_model:
        YieldRegressor.Train()
        YieldRegressor.Test()

    # test_input = np.array([])
