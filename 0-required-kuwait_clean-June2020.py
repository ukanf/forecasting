"""by Felipe Ukan - (c) 2019 Lakes Env. Software
Converts Kuwait file type to our local supported file format

"""
import pandas as pd
from models.utils import utils, json
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import os
# os.path.join('');
# need to list the folder datasets/Kuwait/Active/localformat/raw
# go through every file changing index...
input_path = os.path.join('datasets', 'Kuwait', 'Active', 'localformat', 'raw')
output_path = os.path.join('datasets', 'Kuwait', 'Active', 'localformat', 'cleaned')
translate_names_path = os.path.join('datasets', 'Kuwait', 'Active', 'localformat', 'code_to_parameter_min.json')
absolute_shared_folder = os.path.join('//winnas5', 'd$', 'ProjectData', 'ML_Forecast', 'KUWAIT')
refined_test_full_path = os.path.join(absolute_shared_folder, 'datasets', 'final')
# for data_file_name in os.listdir(input_path):
is_jahra = False
if is_jahra:
    in_absolute_filename_minmax = os.path.join(refined_test_full_path, 'AMS02_Jahra_minmax.csv')
    in_absolute_filename_dataset = os.path.join(refined_test_full_path, 'AMS02_Jahra_raw_training_data_le2015.csv')

    df = pd.read_csv(in_absolute_filename_dataset, header=0)
    # df = df.set_index([pd.to_datetime(df['Date'] + ' ' + df['Time'])])
    # translate_names = json.loads(open(translate_names_path, 'r').read())
    # df = df.rename(columns=translate_names)

    scaled_param = ["O3", "SO2", "NO2", "CO", "PM10"]
    scaler_df = pd.read_csv(in_absolute_filename_minmax)
    scalerY = MinMaxScaler(feature_range=(0, 1))
    scalersY = {}
    for item in scaled_param:
        scalersY[item] = [[scaler_df[item][0]], [scaler_df[item][1]]]
        # "SO2": [[scaler_df["SO2"][0]], [scaler_df["SO2"][1]]],
        # "NO2": [[scaler_df["NO2"][0]], [scaler_df["NO2"][1]]],
        # "CO": [[scaler_df["CO"][0]], [scaler_df["CO"][1]]],
        # "PM10": [[scaler_df["PM10"][0]], [scaler_df["PM10"][1]]],
    # print(scalersY.get("O3"))

    for item in scaled_param:
        scalerY.fit(scalersY.get(item))
        final_colum = scalerY.inverse_transform(df[item].fillna(0).values.reshape(-1, 1))
        df[item] = final_colum

    utils.write_dataframe_to_file(df, os.path.join(refined_test_full_path, 'resolved_AMS02_Jahra_raw_training_data_le2015.csv'))

        # print(os.path.join(output_path, 'cleaned_' + data_file_name))
        # print(df.head())
        # exit()
else:
    # here we handle other datafiles
    params_to_keep = ['NO', 'NH3', 'O3', 'SO2', 'NO2', 'CO2', 'CH4', 'CO', 'PM10', 'PM2.5']
    # files: AlFahaheel_AMS_Data.csv AlMutlah_AMS_Data.csv
    file_name = 'AlFahaheel_AMS_Data.csv'
    print("working on {}".format(file_name))

    in_absolute_filename_dataset = os.path.join(refined_test_full_path, file_name)
    df = pd.read_csv(in_absolute_filename_dataset, header=0)
    df = df.set_index([pd.to_datetime(df['Date'] + ' ' + df['Time'])])
    del df['Date']
    del df['Time']
    # fix date and time to just datetime
    for item in df.columns.values:
        if item not in params_to_keep:
            del df[item]

    print(df.head())

    utils.write_dataframe_to_file(df, os.path.join(refined_test_full_path,
                                                   'resolved' + file_name))
