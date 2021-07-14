# --- general imports
import copy
import datetime as DT
# import matplotlib as plt
import numpy as np
import os
import pandas as pd
from .utils import utils
pd.set_option('display.max_rows', 50)  # so pandas prints more rows
pd.set_option('display.max_columns', 50)  # so pandas prints more rows

# --- START Functions
def clean_datetime_site(df, read_frequency):
    if read_frequency == '1H' or read_frequency == 'H':
        return clean_datetime_site_hourly(df)
    else:
        return False


def clean_datetime_site_hourly(df):
    """
        Clears data in datetime and site field of a given pandas dataframe.
        site becomes: 'site' containing only the site code
        datetime becomes: 'date' containing only year-month-day
        value is repeated.
        :param df: a pandas dataframe with columns: datetime, site and value
        :return: a modified dataframe with columns: datetime, site and value
    """

    # groupby should sort it already
    new_df = df.groupby(['site', 'datetime', 'poc'])

    # initializing variables
    aval_reading_metadata = ['data_status', 'unit', 'qc', 'method_code', 'mpc', 'mpc_value', 'uncertainty', 'qualifiers', 'lat', 'lon', 'GISDatum', 'elev']
    save_meta_change_unique = { x: [] for x in aval_reading_metadata }
    all_uids_meta_info = {}
    datetime = list()
    monitor_uid = list()
    value = list()

    for leftf, rightf in new_df:
        try:
            # unpacking/converting variables that we will use
            curr_timestamp = pd.Timestamp(
                DT.datetime(int(str(leftf[1])[0:4]), int(str(leftf[1])[4:6]), int(str(leftf[1])[6:8]),
                            int(str(leftf[1])[9:11])))
            curr_uid = str(leftf[0]) + str(int(leftf[2]))

            # initializing dict with the meta info for this uid
            if curr_uid not in all_uids_meta_info.keys():
                all_uids_meta_info[curr_uid] = copy.deepcopy(save_meta_change_unique)

            for curr_meta_reading in aval_reading_metadata:
                try:
                    if np.isnan(rightf[curr_meta_reading].values[0]):
                        # if the number is nan just add it as empty string
                        curr_content = ''
                    else:
                        curr_content = str(rightf[curr_meta_reading].values[0])
                except ValueError and TypeError as e:
                    # just convert to string to make sure we store it
                    # print('Got an error: {}.'.format(e))
                    curr_content = str(rightf[curr_meta_reading].values[0])


                if len(all_uids_meta_info[curr_uid][curr_meta_reading]) > 0:
                    # checks if the last element changed.. not very efficient but thats what we got for now
                    if all_uids_meta_info[curr_uid][curr_meta_reading][-1][1] != curr_content:
                        all_uids_meta_info[curr_uid][curr_meta_reading].append([curr_timestamp, curr_content])
                else:
                    # initializes the first element
                    all_uids_meta_info[curr_uid][curr_meta_reading].append([curr_timestamp, curr_content])

            # adding the variables that change constantly to their arrays
            monitor_uid.append(curr_uid)
            value.append(float(rightf.value))
            datetime.append(curr_timestamp)

        except Exception as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            exit()

    temp_df = pd.DataFrame()
    temp_df['monitor_uid'] = monitor_uid
    temp_df['datetime'] = datetime
    temp_df['value'] = value

    # print(all_uids_meta_info)

    # return pd.DataFrame({'site': sitenum, 'date': date, 'value': df.value}) cant use this shorter version anymore
    # because some issues with dictionary size
    return temp_df, all_uids_meta_info


def reindex_by(df, freq_by, year=None):
    """
    Assumes data is already sorted in the right sequency. Reindexes the data by daily date. So, if a day was missing,
    now it will be added and have nan value :param df: a pandas dataframe with columns: datetime, site and value
    :return: a pandas dataframe with columns: datetime, site and value. However, it reindex the data by date and
    fills missing data (because the date didnt exist) for other columns with NaN value
    """
    global start_year, end_year
    if year:
        start_year = year
        end_year = year

    dates = pd.to_datetime(pd.date_range(str(start_year) + '-01-01 00:00:00', str(end_year) + '-12-31 23:00:00', freq=freq_by))  # keeps only the date part
    df = df.reindex(dates)
    df.index.rename('datetime', inplace=True)
    return df


def separate_site(df):
    """
    Separates each site into a new column (sites were inside one single column, now they get separated, one column
    per site) :param df: a pandas dataframe with columns: datetime, site and value indexed by [sitenum,
    date]. :return: new dataframe with date column and one column for each site named: parameter_sitenumber (size
    varies since depends on the number of sites).
    """
    # logger.info('Separating the sites into new columns')

    # 'unstacks' the site values from the site column and creates the new columns
    new_df = df.copy()

    new_df = new_df.set_index(['monitor_uid', 'datetime']).unstack(level=0)
    # need to 'droplevel' because before that, each site was a sub column of a newly created column
    new_df.columns = new_df.columns.droplevel()
    # renames each column to contain the parameter as the prefix
    new_df.columns = [str(col) for col in new_df.columns]

    return new_df


@utils.timeit
def clean_usepa(clean_parameters):
    root_dir = clean_parameters['C_input_dataset_path']

    for item in utils.generate_input_parameters(clean_parameters['C_states']):
        # initializes new dataframe that will be used to save data
        save_df = pd.DataFrame()
        # unpacking variables
        state, county, year, parameter, read_frequency = item

        # working with file path
        full_in_file_path = os.path.join(root_dir, state, county, str(year) + '_' + str(year), 'frequency_' + read_frequency, parameter + '.csv')

        # ------------------------------
        try:
            df = pd.read_csv(full_in_file_path, skipfooter=1, engine='python', index_col=0)

            if not utils.is_dataframe_ok(df):
                print('Dataframe NOT ok for: {}'.format(full_in_file_path))
                continue
            # ------------------------------
            print('Cleaning: {}'.format(full_in_file_path))

            df, all_uids_meta_info = clean_datetime_site(df, read_frequency)

            df = separate_site(df)

            df = reindex_by(df, read_frequency, year=year)

            # BECAREFUL TO NOT OVERWRITE THINGS
            for uid in df.columns:
                full_out_df_file_path = os.path.join(clean_parameters['C_prefix'], state, county, str(year) + '_' + str(year),
                                              'frequency_' + read_frequency, uid + '_df.csv')
                full_out_metadata_file_path = os.path.join(clean_parameters['C_prefix'], state, county, str(year) + '_' + str(year),
                                              'frequency_' + read_frequency, uid + '_meta.json')
                save_df[parameter] = df[uid]
                utils.write_dataframe_to_file(save_df, full_out_df_file_path)
                missing_percentage = (float(save_df[parameter].isnull().sum()) / float(df.index.size)) * 100
                all_uids_meta_info[uid].update({'missing_percentage': [year, round(missing_percentage, 2)]})
                utils.write_json_to_file({parameter: all_uids_meta_info[uid]}, full_out_metadata_file_path)

        except Exception as e:
            print('Error {} while working with: {}'.format(e, full_in_file_path))
            continue
