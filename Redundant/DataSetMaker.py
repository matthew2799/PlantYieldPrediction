import csv, os, sys
import datetime as dt
import pandas as pd
 

def MakeMasterDataSet():
    weight_data = pd.read_csv('../../images_and_annotations/harvested_plant_data/plant_weight_data.csv')

    # print(weight_data.head(80))


    tray_one   = weight_data.loc[weight_data['tray'] == 31]
    tray_two   = weight_data.loc[weight_data['tray'] == 32]
    tray_three = weight_data.loc[weight_data['tray'] == 33]
    tray_four  = weight_data.loc[weight_data['tray'] == 34]

    trays = list()
    for i in range(31, 35):
        trays.append(weight_data.loc[weight_data['tray'] == i])

    # Do the things in order properly

    # Tray, plant, Image, annotations, weight 

    columns   = ['tray', 'plant', 'image_path', 'pixel_count', 'wet_weight', 'dry_weight']
    base_path = '../../images_and_annotations/PSI_Tray0'

    for tray in trays:
        tray_label = str(int(tray.iloc[0,0]))
        path = base_path + tray_label + '/'
        for index, row in tray.iterrows():
            pos = row['positions']
            annotations = pd.read_csv(path + pos + '/PSI_Tray' + tray_label + pos + '.csv')
        


    
def makeHarvestDataset(self, csv_path):
    
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
    makeHarvestDataset()