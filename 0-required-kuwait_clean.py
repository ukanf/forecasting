"""by Felipe Ukan - (c) 2019 Lakes Env. Software
Converts Kuwait file type to our local supported file format

"""
import pandas as pd
from models.utils import utils, json
from matplotlib import pyplot
import os
# os.path.join('');
# need to list the folder datasets/Kuwait/Active/localformat/raw
# go through every file changing index...
input_path = os.path.join('datasets', 'Kuwait', 'Active', 'localformat', 'raw')
output_path = os.path.join('datasets', 'Kuwait', 'Active', 'localformat', 'cleaned')
translate_names_path = os.path.join('datasets', 'Kuwait', 'Active', 'localformat', 'code_to_parameter_min.json')
for data_file_name in os.listdir(input_path):
    df = pd.read_csv(os.path.join(input_path, data_file_name), header=0)
    df = df.set_index([pd.to_datetime(df['Date'] + ' ' + df['Time'])])
    del df['Date']
    del df['Time']
    # del df['Conc,NH3,1']
    # del df['Conc,For,2']
    # del df['Conc,Ben,2']
    # del df['Conc,Tol,2']
    # del df['Conc,pXy,2']
    # del df['Conc,mXy,2']
    # del df['Conc,123TMB,2']
    # del df['Conc,oXy,2']
    # del df['Conc,Etb,2']
    # del df['Conc,Sty,2']
    # del df['Conc,Cl2,2']
    # del df['Conc,CH4,2']
    # del df['Relative Humidity']
    # del df['Pressure']
    # del df['Global Radiation']
    # del df['Rain']
    # del df['PM2.5 Conc']
    # del df['Conc,Ace,2']
    # del df['Conc,Phe,2']
    # del df['Sky Radiation']
    # maybe remove co2 ?
    # got to translate the units
    translate_names = json.loads(open(translate_names_path, 'r').read())
    df = df.rename(columns=translate_names)

    utils.write_dataframe_to_file(df, os.path.join(output_path, data_file_name))

    # print(os.path.join(output_path, 'cleaned_' + data_file_name))
    # print(df.head())
    # exit()
