"""by Felipe Ukan - (c) 2019 Lakes Env. Software

"""
import pandas as pd
from models.utils import utils, json
import os
import numpy as np
from handle_data.utils import utils as handle_data_utils


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
input_file = os.path.join('datasets', 'US_Brian', 'localformat', 'extracted', 'FINAL',
                          place, 'wheatfield_instant_final_merged.csv')
output_file = os.path.join('datasets', 'US_Brian', 'localformat', 'extracted', 'FINAL',
                           place, 'reindexed_wheatfield_instant_final_merged.csv')
# folder where we want to output the data
final_df = pd.DataFrame()

# *********************************
df = pd.read_csv(input_file, header=0)
df = df.set_index(pd.to_datetime(df['Unnamed: 0']))
del df['Unnamed: 0']
del df.index.name
# print(df.index)
df = reindex_by(df, 'H', year=2018)
# print(df.head(15))
# print(df.tail(15))

# print(final_df.head())

utils.write_dataframe_to_file(df, output_file)
