"""by Felipe Ukan - (c) 2019 Lakes Env. Software
Converts Kuwait file type to our local supported file format

"""
import pandas as pd
from models.utils import utils, json
# from models.utils import json
# from matplotlib import pyplot
import os
# os.path.join('');
# need to list the folder datasets/Kuwait/Active/localformat/raw
# go through every file changing index...
weather_data_file = os.path.join('datasets', 'US_Brian', 'localformat', 'cleaned', 'Surface_Black_Hills_2018.csv')
full_domain_folder = os.path.join('datasets', 'US_Brian', 'localformat', 'cleaned', 'full_domain_-103.8269_43.0815')

output_path = os.path.join('datasets', 'US_Brian', 'localformat', 'merged')
li = []
frame = pd.DataFrame()

for data_file_name in os.listdir(full_domain_folder):
    df = pd.read_csv(os.path.join(full_domain_folder, data_file_name), header=0)
    df = df.set_index(pd.to_datetime(df['Unnamed: 0']))
    del df['Unnamed: 0']
    del df.index.name
    li.append(df)
frame = pd.concat(li, axis=0)
frame = frame.loc[~frame.index.duplicated(keep='first')]
# print(frame.index.get_duplicates().unique()) checks duplicates

df = pd.read_csv(weather_data_file, header=0)
df = df.set_index(pd.to_datetime(df['Unnamed: 0']))
del df['Unnamed: 0']
del df.index.name
del df['SensHeatFlux']
del df['SurfFricVel']
del df['ConvectVelScale']
del df['VertPotTempGrad']
del df['PBL']
del df['SBL']
del df['Monin-ObukhovLen']
del df['SurfRoughLen']
del df['BowenRatio']
del df['Albedo']
del df['RefHeightWsWd']
del df['RefHeightTemp']
del df['PrecipitCode']
del df['PrecipitRate']
del df['RelHumidity']
del df['CloudCover']
del df['DataFlag']
result = pd.concat([frame, df], axis=1, sort=False)

# print(result.head(50))

gen_final_file_name = 'fd_-103.8269_43.0815_weather_Surface_Black_Hills_2018.csv'
utils.write_dataframe_to_file(result, os.path.join(output_path, gen_final_file_name))
