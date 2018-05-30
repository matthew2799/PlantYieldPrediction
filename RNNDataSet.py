import csv
import os
import sys
import datetime as dt

import pandas as pd
from skimage import io, transform


import sys
sys.path.append("..")

from CurrentYield.CurrentYieldEstimator  import PlantWeightModel
from FutureYield.FutureYieldEstimator import PlantYieldPredictor

class GenerateRNNData(object):
    
    def __init__(self, master_data, harvest_data, image_dir):
        
        self.master_data = pd.read_csv(master_data)
        self.harvest_data = pd.read_csv(harvest_data)
        self.image_dir = image_dir

        self.harvest_times, self.harvest_dry_weights, self.harvest_fresh_weights = self.get_harvest_times()
        self.dataset = pd.DataFrame()


    def get_harvest_times(self):
        harvest_times = dict()
        harvest_dry_weights = dict()
        harvest_fresh_weights = dict()

        for index, row in self.harvest_data.iterrows():
            date = dt.datetime.strptime(row.loc['datetime'], '%Y-%m-%d:%H-%M-%S')
            dry_weight = row.loc['dry wt (mg)']
            fresh_weight = row.loc['fresh wt (mg)']
            id = str(int(row.loc['tray'])) + '_' + str(row.loc['position'])
            harvest_times[id] = date
            harvest_dry_weights[id] = dry_weight
            harvest_fresh_weights[id] = fresh_weight
        return harvest_times, harvest_dry_weights, harvest_fresh_weights
        
    def make_dataset(self):
        
        CurrentYieldPredictor = PlantWeightModel(load_model=True, data_path='./data/master_dataset.csv', model_name='current_mlp.pkl')
        FutureYieldPredictor = PlantYieldPredictor(load_model=True, net_name='future_mlp.pkl', data_path='./data/RNN_data.csv')


        # WeightPredictor.Train()
        # WeightPredictor.Test()

        # Dataframe Columns: tray, plant, image_name, 
        for index, row in self.master_data.iterrows():
            # Get Row info
            tray = str(int(row.loc['tray']))
            pos  = str(row.loc['position'])
            image_num  = str(row.loc['image_num'])
            name = str(row.loc['name'])
            date = str(row.loc['datetime'])

            # Get harvest info
            id = tray + '_' + pos
            harvest_date = self.harvest_times[id]
            target_dry   = self.harvest_dry_weights[id]
            target_fresh = self.harvest_fresh_weights[id]

            # Get Weight Prediction
            path = os.path.join(self.image_dir, 'PSI_Tray0' + tray, pos, name)
            image = io.imread(path)
            current_yield = CurrentYieldPredictor.Guess(image)
            print(current_yield)
            current_date = dt.datetime.strptime(date, '%Y-%m-%d:%H-%M-%S')
            

            # hth, rem = divmod((harvest_date - current_date).total_seconds(), 3600) # Hours Until Harvest
            dth = (harvest_date - current_date).days
            future_yield = FutureYieldPredictor.Guess(current_yield, dth)
            has = (56 - dth)         # Days After Sewing
            new_row = pd.Series([tray, pos, name, current_yield, date, dth, has, target_dry, target_fresh, future_yield,image_num], 
                                index=['tray','pos','name','current_yield','date','DTH', 'HAS', 'target_dry', 'target_fresh', 'future_yield', 'image_num'])

            self.dataset = self.dataset.append(new_row, ignore_index=True)
        self.dataset.to_csv('data/TrainingSet.csv') 

if __name__ == "__main__":

    master    = 'data/master_dataset.csv'
    harvest   = 'data/harvested.csv'
    image_dir = '../images_and_annotations/'

    data_gen = GenerateRNNData(master, harvest, image_dir);
    data_gen.make_dataset()

