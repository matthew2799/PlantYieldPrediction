import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.init as init
from   torch.autograd import Variable

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import mean_absolute_error

from CNNDataLoader import DataSplitter, CNNDataSet, SquareRescale, ToTensor


# Hyper Parameters
num_epochs = 20
batch_size = 1
learning_rate = 0.001

# DataLoaders
print ('Loading model data...')

data_path = '../../data/harvested.csv'
image_dir = '../../../images_and_annotations'


splitter = DataSplitter()
train, test  = splitter(r'E:\Documents\Engineering\THESIS\ThesisProject\data/harvested.csv')

# MNIST Dataset
train_dataset = CNNDataSet( train,
                            image_dir='../../../images_and_annotations/',
                            transform=transforms.Compose([SquareRescale(200),
                                                            ToTensor()]))

test_dataset = CNNDataSet(  test,
                            image_dir='../../../images_and_annotations/',
                            transform=transforms.Compose([SquareRescale(200),
                                                            ToTensor()]))
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

class PlantWieghtRegressor(nn.Module):
    def __init__(self, batch_size=10):
        super(PlantWieghtRegressor, self).__init__()
        self.batch_size=batch_size
        self.conv1 = nn.Conv2d(3, 5,  kernel_size=2)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=2)
        self.mp = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(24010, 5000)
        self.fc2 = nn.Linear(5000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 1)

        fc = [self.fc1, self.fc2, self.fc3, self.fc4]
        self.relu = nn.ReLU()

    def forward(self, x):
        in_size = x.size(0)
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
        return x.view(self.batch_size)

        

train_model = True
gpu_available = torch.cuda.is_available()

net = PlantWieghtRegressor(batch_size=batch_size)

if gpu_available:
    net = net.cuda()
    print('Using GPU...')
else:
    print('Using CPU...')

if train_model:
    print('Training the model ...')
    
    # Loss and Optimizer
    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            images = Variable(data['image']).float()
            targets = Variable(data['target']).float()

            if gpu_available:
                images  = images.cuda()
                targets = targets.cuda()

            output = net(images)    

            # print('Output', output)
            # print('Target', target)
            loss = criterion(output, targets) 

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_labels        = targets.cpu().numpy() if gpu_available else targets.numpy()
        y_pred_results  = output.cpu().data.numpy() if gpu_available else output.data.numpy()

        error = mean_absolute_error(y_labels, y_pred_results);

        print ('Epoch: {}, MA Error {:.2f}, MSE Loss {:.2f}'.format(epoch, error, loss.data[0]))

    net.eval()    # Change model to 'eval' mode (BN uses moving mean/var).

else:
    net.load_state_dict(torch.load('cnn_predict.pkl'))


# Test the Model
mean_error = 0
total_error = 0
num_batches = 0
for data in test_loader:
    images = Variable(data['image']).float()
    targets = Variable(data['target']).float()

    if gpu_available:
        images = images.cuda()
        labels = targets.cuda()

    outputs = net(images)

    y_labels = targets.cpu().numpy() if gpu_available else targets.numpy()
    y_pred_results = outputs.cpu().data.numpy() if gpu_available else y_labels.data.numpy()

    mean_error = mean_absolute_error(y_labels, y_pred_results)
    print(y_labels)
    print(y_pred_results)
    num_batches += 1
    total_error += mean_error 


print('Test Accuracy of the model on the 10000 test images: %d %%' % ( total_error / num_batches))

# Save the Trained Model
torch.save(net.state_dict(), 'cnn.pkl')

def in_range(prediction, truth, lower, upper):
    
    if prediction > lower*truth and prediction < upper*truth:
        return True
    else:
        return False


