"""by Felipe Ukan - (c) 2019 Lakes Env. Software
Extracts gas data from huge files given a vector o latlongs

"""
import pandas as pd
from models.utils import utils, json
import os

# folder where all the fullDomain files are
full_domain_input_path = os.path.join('C://', 'huge_datasets', 'Wheatfield_Instant')
# folder where we want to output the data
output_path = os.path.join('datasets', 'US_Brian', 'localformat', 'extracted', 'Wheatfield_Instant')

# just a file that we use to translate the variable names to a shorter name
translate_names_path = os.path.join('datasets', 'US_Brian', 'localformat', 'code_to_parameter_min.json')

# array of tuples that will get data extracted
# IMPORTANT -- ROUND TO 5 DECIMALS
lat_long_to_extract = [(-105.0265,42.00221), (-104.9782,42.04725), (-104.948,42.05625), (-104.9661,42.11029),
                       (-104.9117,42.07426), (-105.0567,42.10128)]

# ********************************* EXTRACTS
for data_file_name in os.listdir(full_domain_input_path):
    # loads files
    translate_names = json.loads(open(translate_names_path, 'r').read())
    df = pd.read_csv(os.path.join(full_domain_input_path, data_file_name), header=0)
    print(os.path.join(full_domain_input_path, data_file_name))

    # sets time as index and deletes the column (keeping the index)
    df = df.set_index(pd.to_datetime(df['time']))
    del df['time']
    del df.index.name

    # translates to new var names
    df = df.rename(columns=translate_names)

    # groups the whole file into latlongs and iterates over them
    new_df = df.groupby(['x', 'y'])
    count = 0
    for leftf, rightf in new_df:
        # have to convert to 5 decimals each field of the tuple
        leftf = (round(leftf[0], 5), round(leftf[1], 5))
        # if the x,y tuple is in the array we extract data

        if leftf in lat_long_to_extract:
            temp_df = pd.DataFrame()
            temp_df['conc_tracer'] = rightf["ConcTracer"]
            temp_df['dep_tracer'] = rightf["DepTracer"]
            temp_df['datetime'] = rightf.index
            temp_df.set_index('datetime', inplace=True)
            del temp_df.index.name
            df = temp_df
            now_folder_name = str(leftf)

            # saves individual files
            utils.write_dataframe_to_file(df, os.path.join(output_path, now_folder_name, 'individual_files',
                                                           data_file_name))
            print(os.path.join(output_path, 'cleaned_' + data_file_name))

# ********************************* MERGES
# second part of the script where we merge all the individual files into one "merged" file per latlong tuple
full_domain_folder = output_path
li = []
frame = pd.DataFrame()

# goes over all individual files
try:
    for data_folder_name in os.listdir(full_domain_folder):
        path_for_individual_files = os.path.join(full_domain_folder, data_folder_name, 'individual_files')
        for data_file_name in os.listdir(path_for_individual_files):
            df = pd.read_csv(os.path.join(path_for_individual_files, data_file_name), header=0)
            # we set the index to be the datetime (which is unamed from the previous dataframe)
            df = df.set_index(pd.to_datetime(df['Unnamed: 0']))
            del df['Unnamed: 0']
            del df.index.name
            li.append(df)

        # we concatenate and remove the duplicate values keeping the last valid entry
        frame = pd.concat(li, axis=0, sort=False)
        frame = frame.loc[~frame.index.duplicated(keep='last')]

        utils.write_dataframe_to_file(frame, os.path.join(output_path, data_folder_name, 'merged.csv'))
except Exception as e:
    print('Exception: {}'.format(e))
