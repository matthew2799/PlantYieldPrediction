import sys
import imghdr
import os
import pandas as pd


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


from CurrentYield.CurrentYieldEstimator import PlantWeightModel
from FutureYield.FutureYieldEstimator   import PlantYieldPredictor


def train_mlp_future():
    
    batch_size = 200
    num_epochs = 20
    optimizer_params =  {   'learning_rate' : 0.00015,
                            'momentum' : 0.9,
                            'decay' : 0.5,
                            'amsgrad' : False}

    hidden_size=[1000,500,100,10]

    FutureYieldModel = PlantYieldPredictor(net_name='future_mlp.pkl', load_model=False, 
                                            save_model=True, hidden_size=hidden_size,
                                            num_epochs=num_epochs, batch_size=batch_size, 
                                            optimizer_params=optimizer_params)

    train_results = FutureYieldModel.Train()
    test_results = FutureYieldModel.Test()
    
    # Plot the results in matplotlib

    mse_history = train_results['mse_history']
    mae_history = train_results['mae_history']
    num_epochs = train_results['num_epochs']

    x_range = range(num_epochs)
    
    fig, ax1 = plt.subplots()
    ax1.plot(x_range, mse_history, 'b-')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel('Mean Squared Error Loss')
    ax1.set_title('Training Error for')

    ax2 = ax1.twinx()
    ax2.plot(x_range, mae_history, 'r-')
    ax2.set_ylabel('Mean Absolute Error')

    fig.tight_layout()
    plt.show()


def train_cnn_current():
 
    batch_size = 5
    num_epochs = 20
    optimizer_params  = {   'learning_rate' : 0.015,
                            'momentum' : 0.9,
                            'decay' : 2,
                            'amsgrad' : False}

    FutureYieldModel = PlantWeightModel(optimizer_params, model_name='current_mlp.pkl', load_model=False, save_model=True,
                                            num_epochs=num_epochs, batch_size=batch_size)

    train_results = FutureYieldModel.Train()
    # test_results = FutureYieldModel.Test()
    
    # Plot the results in matplotlib

    mse_history = train_results['mse_history']
    mae_history = train_results['mae_history']
    num_epochs = train_results['num_epochs']

    x_range = range(num_epochs)
    
    fig, ax1 = plt.subplots()
    ax1.plot(x_range, mse_history, 'b-')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel('Mean Squared Error Loss', color='b')
    ax1.set_title('Training Error - CNN')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(x_range, mae_history, 'r-')
    ax2.set_ylabel('Mean Absolute Error', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # train_cnn_current()
    train_mlp_future()
