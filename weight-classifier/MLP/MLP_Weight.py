import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.init as init
from   torch.autograd import Variable

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import mean_absolute_error

from MLPDataLoader import DataSplitter, MLPDataSet


# Hyper Parameters
num_epochs = 10
batch_size = 10
learning_rate = 0.15


# Model Parameters
input_layer  = 5   
hidden_layers = [100, 100, 50, 20]
output_layer = 1


print ('Loading model data ...')

splitter = DataSplitter()
train, test  = splitter(r'E:\Documents\Engineering\THESIS\ThesisProject\data/harvested.csv')

train_dataset = MLPDataSet(train)
test_dataset  = MLPDataSet(test)


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


class PlantWieghtRegressor(nn.Module):
    
    def __init__(self, input_layer=1, output_layer=1, hidden_layers=[20, 100, 100, 20]):
        super(PlantWieghtRegressor, self).__init__()
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
    

train_model = True
gpu_available = torch.cuda.is_available()

net = PlantWieghtRegressor(input_layer=input_layer, output_layer=output_layer, hidden_layers=hidden_layers)
net.construct_layers()

if gpu_available:
    net = net.cuda()
    print('Using GPU ...')
else:
    print('Using CPU ...')

if train_model:
    print('Training the model ...')
    
    # Loss and Optimizer
    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
            for i, data in enumerate(train_loader):
                annotation = Variable(data['annotation']).float()
                target = Variable(data['target']).float()
                if gpu_available:
                    annotation = annotation.cuda()
                    target =     target.cuda()

                output = net.model(annotation)    

                # print('Output', output)
                # print('Target', target)
                loss = criterion(output, target) 


                # Forward + Backward + Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            y_labels = target.cpu().numpy() if gpu_available else target.numpy()
            y_pred_results = output.cpu().data.numpy() if gpu_available else output.data.numpy()

            error = mean_absolute_error(y_labels, y_pred_results);

            print ('Epoch: {}, MA Error {:.2f}, MSE Loss {:.2f}'.format(epoch, error, loss.data[0]))


