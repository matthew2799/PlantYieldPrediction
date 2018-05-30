import csv
import os
import sys
import datetime as dt
import pandas as pd

def MakeMasterDataSet(path='../../images_and_annotations/harvested_plant_data/plant_weight_data.csv'):
    
    weight_data = pd.read_csv(path)

    trays = list()
    for i in range(31, 35):
        trays.append(weight_data.loc[weight_data['tray'] == i])


    plant_data = pd.read_csv(
        '../../images_and_annotations/PSI_Tray031/p-1/PSI_Tray031p-1.csv')
    base_path = '../../images_and_annotations/PSI_Tray0'

    master_frame = pd.DataFrame()

    # for tray in trays:
    for tray in trays:
        tray_label = str(int(tray.iloc[0, 0]))
        path = base_path + tray_label + '/'
        for tray_index, tray_row in tray.iterrows():
            pos = tray_row['position']
            annotations = pd.read_csv(
                path + pos + '/PSI_Tray0' + tray_label + pos + '.csv')

            for index, row in annotations.iterrows():
                new_row = row.append(tray_row)
                master_frame = master_frame.append(new_row, ignore_index=True)

    master_frame.to_csv('master_dataset.csv')

    
def makeHarvestDataset(csv_path):
    
    data = pd.read_csv(csv_path)

    end_data = pd.DataFrame()

    trays  = list()
    for i in range(31, 35):
        tray = data.loc[data['tray'] == i]

        for j in range(1, 21):
            label = 'p-' + str(j)
            pos = tray.loc[tray['position'] == label]
            print(len(pos))
            max_row = pos['image_num'].argmax()

            print('Pos:', len(pos), "Row:", max_row)

            new_row = data.iloc[[max_row]]
            end_data = end_data.append(new_row, ignore_index=True)        

    end_data.to_csv('../../data/harvested.csv')

if __name__ == '__main__':
    MakeMasterDataSet()
    makeHarvestDataset('master_dataset.csv')