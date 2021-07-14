"""by Felipe Ukan - (c) 2019 Lakes Env. Software

"""
import pandas as pd
from models.utils import utils, json
import os
import numpy as np

def reindex_by(df, freq_by, year=None):
    """
    Assumes data is already sorted in the right sequency. Reindexes the data by daily date. So, if a day was missing,
    now it will be added and have nan value :param df: a pandas dataframe with columns: datetime, site and value
    :return: a pandas dataframe with columns: datetime, site and value. However, it reindex the data by date and
    fills missing data (because the date didnt exist) for other columns with NaN value
    """
    start_year = year
    end_year = year

    dates = pd.to_datetime(pd.date_range(str(start_year) + '-01-01 00:00:00', str(end_year) + '-12-31 23:00:00', freq=freq_by))  # keeps only the date part
    df = df.reindex(dates, fill_value=-35)
    df.index.rename('datetime', inplace=True)
    return df

# folder where all the fullDomain files are
place = 'Wheatfield_Instant'
conc_input_path = os.path.join('datasets', 'US_Brian', 'localformat', 'extracted', place)
weather_input_path_file = os.path.join('datasets', 'US_Brian', 'localformat', 'extracted', 'MMIF_AERCOARE',
                                       place, 'weather_merged.csv')
# folder where we want to output the data
output_path = os.path.join('datasets', 'US_Brian', 'localformat', 'extracted', 'FINAL', place)
collapse_to_1h_avg = True
lowest_value = -35

final_df = pd.DataFrame()

# *********************************
for data_folder_name in os.listdir(conc_input_path):
    conc_input_path_file = os.path.join(conc_input_path, data_folder_name, 'merged.csv')
    # loads file
    df = pd.read_csv(conc_input_path_file, header=0, index_col=0)
    print(conc_input_path_file)
    nice_prefix = data_folder_name.replace('(', '').replace(')', '').replace(' ', '')
    df.rename(columns=lambda x: x + '_' + nice_prefix, inplace=True)
    if collapse_to_1h_avg:
        # here we have a moving window that takes the average of the current and 3 previous values to calculate the field
        # this is to calculate the average for the hour
        df['conc_tracer_'+nice_prefix+'_rolling4back'] = df['conc_tracer_'+nice_prefix].rolling(4).mean()
        df['dep_tracer_' + nice_prefix + '_rolling4back'] = df['dep_tracer_' + nice_prefix].rolling(4).mean()
    for col in df.columns:
        #
        df[col] = np.log(np.multiply(np.power(10, 20), df[col]))
    final_df = pd.concat([final_df, df], axis=1, sort=False).replace(np.NINF, lowest_value)

weather_df = pd.read_csv(weather_input_path_file, header=0, index_col=0)
final_df = pd.concat([weather_df, final_df], axis=1, sort=False)
final_df.set_index(pd.to_datetime(final_df.index))
# final_df = reindex_by(final_df, 'H', year=2018)
final_df.fillna(lowest_value, inplace=True)
# print(final_df.head(50))
utils.write_dataframe_to_file(final_df, os.path.join(output_path, 'final_merged_reindexed.csv'))
