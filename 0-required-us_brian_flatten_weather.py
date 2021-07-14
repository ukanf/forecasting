"""by Felipe Ukan - (c) 2019 Lakes Env. Software

"""
import pandas as pd
from models.utils import utils, json
import os
import numpy as np

# folder where all the fullDomain files are
full_domain_input_path = os.path.join('C://', 'huge_datasets', 'MMIF_AERCOARE', 'Wheatfield')
# folder where we want to output the data
output_path = os.path.join('datasets', 'US_Brian', 'localformat', 'extracted', 'MMIF_AERCOARE', 'Wheatfield')
convert_wd_to_sincos = True

final_df = pd.DataFrame()

# *********************************
for data_file_name in os.listdir(full_domain_input_path):
    # loads file
    df = pd.read_csv(os.path.join(full_domain_input_path, data_file_name), header=0)
    print(os.path.join(full_domain_input_path, data_file_name))
    df.columns = df.columns.str.replace(' ', '')  # deletes all white spaces from columns
    df = df.set_index([pd.to_datetime(dict(year=df['yr'], month=df['mo'], day=df['dy'], hour=df['hr']))])
    del df['yr']
    del df['mo']
    del df['dy']
    del df['hr']
    del df['tsea']
    del df['relh']
    del df['pres']
    del df['srad']
    del df['rdow']
    del df['rain']
    del df['tsky']
    del df['mixh']
    del df['vptg']
    del df['zwsp']
    del df['ztem']
    del df['zrel']
    del df['zdep']

    if convert_wd_to_sincos:
        df['wdir_sin'] = np.sin(np.deg2rad(df['wdir']))
        df['wdir_cos'] = np.cos(np.deg2rad(df['wdir']))
        del df['wdir']

    nice_prefix = data_file_name.split('_AERCOARE')[0]
    df.rename(columns=lambda x: x + '_' + nice_prefix, inplace=True)
    final_df = pd.concat([final_df, df], axis=1, sort=False)

utils.write_dataframe_to_file(final_df, os.path.join(output_path, 'weather_merged.csv'))
