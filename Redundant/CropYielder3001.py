import torch
import torch.utils.data as data_utils
import torch.nn.init as init
from torch.autograd import Variable

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import mean_absolute_error
from sklearn.externals.joblib import Memory
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline

import numpy as np
import pandas as pd

from PlantImageDataset import PlantImageDataset, SquareRescale, ToTensor


# Temp imports
import datetime
from dateutil.parser import parse
import inspect


class RegressionModel(BaseEstimator, RegressorMixin):

    def __init__(self, output_dim=1, import_dim=100, hidden_layer_dims=[100 100],
                 num_epochs, learning_rate, batch_size=100, shuffle=False, callbacks=[]
                 cude_enable=False)

        self.output_dim = output_dim
        self.import_dim = import_dim
        self.hidden_layer_dims = hidden_layer_dims
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.gpu = cude_enable and torch.cuda.is_available()

    def construct_model(self):