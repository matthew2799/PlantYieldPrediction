import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from PlantImageDataset import PlantImageDataset, SquareRescale, ToTensor
from torch.autograd import Variable


from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import mean_absolute_error



# Hyper Parameters
num_epochs = 5
batch_size = 10
learning_rate = 0.001

# DataLoaders
print ('Loading model data...')

# MNIST Dataset
train_dataset = PlantImageDataset(csv_file='../images_and_annotations/PSI_Tray031/p-2/PSI_Tray031p-2.csv',
                                    image_dir='../images_and_annotations/PSI_Tray031/p-2',
                                    transform=transforms.Compose([SquareRescale(445),
                                                                 ToTensor()]))

test_dataset = PlantImageDataset(csv_file='../images_and_annotations/PSI_Tray031/p-4/PSI_Tray031p-4.csv',
                                    image_dir='../images_and_annotations/PSI_Tray031/p-4',
                                    transform=transforms.Compose([SquareRescale(445),
                                                                 ToTensor()]))
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 5,  kernel_size=2)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=2)
        self.mp = nn.MaxPool2d(2)

        self.fc = nn.Linear(121000, 1)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.mp(F.relu(self.conv2(x)))
        x = x.view(in_size, -1)
        x = self.fc(x)

train = False
gpu_available = torch.cuda.is_available()


cnn = CNN()
if gpu_available:
    cnn = cnn.cuda()
    print('Using GPU ...')
else:
    print('Using CPU ...')


if train:
    print('Training the model ...')
    # Loss and Optimizer
    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            images = Variable(data['image']).float()
            labels = Variable(data['labels']).float()
            
            if gpu_available:
                images = images.cuda()
                labels = labels.cuda()

            outputs = cnn(images)    
            loss = criterion(outputs, labels) 


            # Forward + Backward + Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_labels = labels.cpu().numpy() if gpu_available else labels.numpy()
        y_pred_results = outputs.cpu().data.numpy() if gpu_available else y_labels.data.numpy()

        error = mean_absolute_error(y_labels, y_pred_results);

        print ('Epoch: %d, Loss %.4f' % (epoch, loss.data[0]))

    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
else: 
    cnn.load_state_dict(torch.load('cnn.pkl'))

# Test the Model
mean_error = 0
total_error = 0
num_batches = 0
for data in test_loader:
    images = Variable(data['image']).float()
    labels = Variable(data['annotation']).float()

    if gpu_available:
        images = images.cuda()
        labels = labels.cuda()

    outputs = cnn(images)

    y_labels = labels.cpu().numpy() if gpu_available else labels.numpy()
    y_pred_results = outputs.cpu().data.numpy() if gpu_available else y_labels.data.numpy()

    mean_error = mean_absolute_error(y_labels, y_pred_results)
    print(y_labels)
    print(y_pred_results)
    num_batches += 1
    total_error += mean_error 

    

print('Test Accuracy of the model on the 10000 test images: %d %%' % ( total_error / num_batches))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')

def in_range(prediction, truth, lower, upper):
    
    if prediction > lower*truth and prediction < upper*truth:
        return True
    else:
        return False
