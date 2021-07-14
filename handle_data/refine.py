"""
    This file constains methods to interpolate and handle with some outliers.
"""
# --- general imports
# import datetime as DT
# import matplotlib as plt
import json
import logging
import time

import numpy as np
import os
import pandas as pd
from .utils import utils
import data_retriever as dr

pd.set_option('display.max_rows', 200000)
pd.set_option('display.max_columns', 20)


# # --- START Functions
def handle_outliers(df, options, unique_identifier):
    """
    Unfortunately here we can have several different states for clipping a series
    :param df:
    :param options:
    :return:
    """
    logger = logging.getLogger(unique_identifier)
    upper_percentile_clip = 98
    # upper_abs_clip = False
    # lower_percentile_clip = False
    lower_abs_clip = 0
    try:
        if 'R_o_upper_clip' in options:
            upper_percentile_clip = options['R_o_upper_clip']['percentile'] if 'percentile' in options[
                'R_o_upper_clip'] else upper_percentile_clip
            # upper_abs_clip = options['R_o_upper_clip']['absolute'] if 'absolute' in options['R_o_upper_clip'] else
            # upper_abs_clip
        if 'R_o_lower_clip' in options:
            # lower_percentile_clip = options['R_o_lower_clip']['percentile'] if 'percentile' in options[
            # 'R_o_lower_clip'] else lower_percentile_clip
            lower_abs_clip = options['R_o_lower_clip']['absolute'] if 'absolute' in options[
                'R_o_lower_clip'] else lower_abs_clip

        upp = np.percentile(df.dropna().values, upper_percentile_clip)
        lpp = lower_abs_clip
        # lpp = np.percentile(df.values, lower_percentile_clip)
        df = df.clip(upper=upp, lower=lpp)

    except Exception as e:
        logger.critical('ERROR: {} while handling outliers'.format(e))

    return df


def handle_divide_wind_sectors(df, refine_parameters):
    for col in df.columns:
        try:
            options = refine_parameters['R_selected_param_station'][col]
            if 'R_o_divide_wind_sectors' in options:
                wind_sectors_parameters = options['R_o_divide_wind_sectors']
                df = df.join(resolve_divide_wind_sectors(df[col], wind_sectors_parameters, col))
        except KeyError as e:
            print('ERROR: {} while handling wind sector'.format(e))
            continue

    return df


def resolve_divide_wind_sectors(df, wind_sectors_parameters, wind_param):
    try:
        sectors_to_divide = wind_sectors_parameters['num_sectors']
        if 360 % sectors_to_divide != 0:
            print('ERROR: divide wind sectors not divisible by 360')
            return df

        new_df = pd.DataFrame()
        new_df['sectors_' + wind_param] = pd.cut(df, sectors_to_divide, labels=[x for x in range(sectors_to_divide)])
        # adds one hot encoding to the dataframe
        new_df = new_df.join(pd.get_dummies(new_df, prefix='wind_sector'))

        return new_df
    except Exception as e:
        print('Error: {} while resolving wind sectors'.format(e))
        return df


def handle_interpolate(df, col_name, param_options, unique_identifier):
    """
    Calculates dispersion of missing data and interpolates for small and medium gaps. Also sets True to a flag if it
    was not possible to fill missing values with small and medium methods. :param df: pandas dataframe with date and
    other values from parameters :return: tuple with: modified dataframe or not and a flag saying if the given
    column/dataframe should be deleted or not (True deletes the column because there is not enough data for
    interpolate and False means the column contains data for all days)
    """
    logger = logging.getLogger(unique_identifier)
    new_df = df.isnull().astype(int).groupby(
        df.notnull().astype(int).cumsum()).sum()

    try:
        max_gap_size_allowed = param_options[
            'R_o_maximum_gap_size'] if 'R_o_maximum_gap_size' in param_options.keys() else new_df.max().astype(int)

        # size of linear interpolation
        linear_interpolate_gaps_size = param_options[
            'R_o_linear_interpolation'] if 'R_o_linear_interpolation' in param_options.keys() else 6
        # size of gap (depends on how data is on the input.. if 1h then 24 gap would be 1 day)
        cycle_size = param_options['R_o_cycle_size'] if 'R_o_cycle_size' in param_options.keys() else 24
        # size of window to look for values (will go forward and backwards this number of times)
        number_of_cycles = param_options[
            'R_o_number_of_cycles'] if 'R_o_number_of_cycles' in param_options.keys() else 1
        # maximum consecutive days/gaps without readings
        largest_gap_size_possible = new_df.max().astype(int)

        if largest_gap_size_possible > max_gap_size_allowed:
            # will return with True because this column will be dropped!
            logger.warning('Largest gap in {} is ({}) bigger than the largest gap allowed ({})'.format(col_name,
                                                                                              largest_gap_size_possible,
                                                                                              max_gap_size_allowed))
            return df

        # linear interpolation
        df = df.interpolate(limit_direction='forward', limit=linear_interpolate_gaps_size)

        # interpolation for average from next and previous cycles
        # df[df.isnull()] = np.nanmean([df.shift(x).values for x in -- uses past data!!
        #                               range(-cycle_size * number_of_cycles, cycle_size * (number_of_cycles + 1),
        #                                     cycle_size)], axis=0)
        df[df.isnull()] = np.nanmean([df.shift(x).values for x in
                                      range(0, cycle_size * (number_of_cycles + 1),
                                            cycle_size)], axis=0)

        new_df = df.isnull().astype(int).groupby(df.notnull().astype(int).cumsum()).sum()

        if new_df.max().astype(int) > 0:
            # means that there is not enough data from previous or next cycle_size to help fit this point, i.e.
            # there is still nan values in the dataframe
            logger.warning('Cannot interpolate data for {}'.format(col_name))
            logger.warning('Given cycle is not sufficient to cover the largest gap of {}'.format(largest_gap_size_possible))
            return df

        return df

    except Exception as e:
        logger.critical('CRITICAL Error: {}'.format(e))
        logger.critical('Dropping column {}'.format(col_name))
        return df

def drop_24h_blocks_with_nan(df, col, unique_identifier):
    """
    :param df:
    :param col:
    :param unique_identifier:
    :return:
    """
    logger = logging.getLogger(unique_identifier)

    blocks_24h = df.groupby(df.index.date)
    valid_values = np.array([])
    valid_dates = []
    for leftf, rightf in blocks_24h:
        # for x in rightf.index.values:
        #     valid_dates.append(x)
        for x in rightf.index.values:
            valid_dates.append(x)
        if np.isnan(rightf.values).any():
            valid_values = np.append(valid_values, [np.nan for x in rightf.values])
            continue
        # print(rightf.values)
        # print(rightf.index.values)
        valid_values = np.append(valid_values, rightf.values)

        # print(leftf)
        # print(rightf)
    # print(valid_values)
    # print(valid_dates)
    df = pd.DataFrame({col: valid_values}, index=valid_dates)
    # print(df.head(800))
    return df


def drop_24h_blocks_with_sequence_of_zeros(df, col, unique_identifier, refine_parameters):
    logger = logging.getLogger(unique_identifier)

    blocks_24h = df.groupby(df.index.date)
    valid_values = np.array([])
    valid_dates = []
    min_nonzeros_in_day = refine_parameters['R_min_nonzeros_in_day']
    for leftf, rightf in blocks_24h:
        for x in rightf.index.values:
            valid_dates.append(x)
        #  todo might want to change check seq of zeros inside right.values(?)
        #  if the number of non zero numbers is less than min_nonzeros_in_day we nan the whole day and drop later
        if np.count_nonzero(rightf.values) < min_nonzeros_in_day:
            valid_values = np.append(valid_values, [np.nan for x in rightf.values])
            continue
        valid_values = np.append(valid_values, rightf.values)

    df = pd.DataFrame({col: valid_values}, index=valid_dates)
    # print(df.head(800))
    return df



@utils.timeit
def refine_usepa_dataset(refine_parameters):
    root_dir = refine_parameters['R_input_dataset_path']

    country = refine_parameters['R_country_code']
    state = refine_parameters['R_state_code']
    all_missing_percentage = dr.get_all_missing_percentage_info_usepa(refine_parameters)
    run_id = str(int(time.time()))

    for df, all_metainfo, country, state, county, station in dr.get_station_data_usepa(refine_parameters, country,
                                                                                        state):

        df.set_index(pd.to_datetime(df.index.values), inplace=True)
        df_final = pd.DataFrame(index=df.index.values)
        station_id = country + state + county + station
        unique_identifier = station_id
        utils.setup_logger(unique_identifier, os.path.join(refine_parameters["R_prefix"], run_id, 'logs', unique_identifier + '.log'), logging.DEBUG)
        logger = logging.getLogger(unique_identifier)
        logger.debug('Starting Logger')

        for col in df.columns:
            # parameter, poc = col.split('-')

            df_onecol_values = pd.DataFrame(df[col].loc[df[col].first_valid_index():df[col].last_valid_index()])

            # interpolates data
            df_onecol_values[col] = handle_interpolate(df_onecol_values[col], col, refine_parameters, unique_identifier)

            # remove outliers
            df_onecol_values[col] = handle_outliers(df_onecol_values[col], refine_parameters, unique_identifier)

            # sets nan to 24h blocks that have at least 1 nan
            df_onecol_values[col] = drop_24h_blocks_with_nan(df_onecol_values[col], col, unique_identifier)

            # df = handle_divide_wind_sectors(df, refine_parameters)

            df_final = pd.concat([df_final, df_onecol_values], axis=1, sort=False)

        full_df_out_file_path = os.path.join(refine_parameters["R_prefix"], run_id, 'datasets', unique_identifier + '.csv')
        full_info_out_file_path = os.path.join(refine_parameters["R_prefix"], run_id, 'metadata', unique_identifier + '.json')

        df_final.rename(columns=lambda x: x + '-' + 'refined', inplace=True)
        df.rename(columns=lambda x: x + '-' + 'original', inplace=True)
        df_final = pd.concat([df_final, df], axis=1, sort=False)

        utils.write_dataframe_to_file(df_final, full_df_out_file_path)
        utils.write_json_to_file(all_metainfo, full_info_out_file_path)

        logger.removeHandler(logger)

        # print(df_final.head(1000))
        # print(df_final.tail(50))
        # print(df_final.shape)
        # exit()

    return


@utils.timeit
def refine_kuwait_dataset(refine_parameters):
    # iterate through files in the given folder
    # for every file go through the columns
    run_id = str(int(time.time()))
    unique_identifier = run_id

    for df, file in dr.get_dataframe_kuwait(refine_parameters):
        # print(df.head())

        df.set_index(pd.to_datetime(df.index.values), inplace=True)
        df_final = pd.DataFrame(index=df.index.values)
        # df = df.replace(np.nan, 0)



        for col in df.columns:
            # here I had to removed the first and last valid index (even though they speed things up) because we were
            # getting a NaT on the first row....
            df_onecol_values = pd.DataFrame(df[col])

            # remove outliers
            df_onecol_values[col] = handle_outliers(df_onecol_values[col], refine_parameters, unique_identifier)

            # interpolates data
            df_onecol_values[col] = handle_interpolate(df_onecol_values[col], col, refine_parameters, unique_identifier)

            # sets nan to 24h blocks that have at least 1 nan
            df_onecol_values[col] = drop_24h_blocks_with_nan(df_onecol_values[col], col, unique_identifier)

            # sets nan to 24h blocks that do not have enough R_min_nonzeros_in_day (non zeros in a day)
            df_onecol_values[col] = drop_24h_blocks_with_sequence_of_zeros(df_onecol_values[col], col,
                                                                           unique_identifier, refine_parameters)

            # saves in final df
            df_final = pd.concat([df_final, df_onecol_values], axis=1, sort=False)

        full_df_out_file_path = os.path.join(refine_parameters["R_output_dataset_path"],
                                             unique_identifier + '-' + file)

        df_final.rename(columns=lambda x: x + '-' + 'refined', inplace=True)
        # df.rename(columns=lambda x: x + '-' + 'original', inplace=True)
        # df_final = pd.concat([df_final, df], axis=1, sort=False)

        utils.write_dataframe_to_file(df_final, full_df_out_file_path)

    return


@utils.timeit
def refine_usbrian_dataset(refine_parameters):
    return
