"""by Felipe Ukan - (c) 2019 Lakes Env. Software
Converts Kuwait file type to our local supported file format

"""
import pandas as pd
from models.utils import utils, json
import os

# os.path.join('');
# need to list the folder datasets/Kuwait/Active/localformat/raw
# go through every file changing index...
#
files_full_domain = ["example.csv"]
# files_full_domain = ["FullDomain_2018_0" + str(x) + ".csv" for x in range(2, 10)]
# files_full_domain.append("FullDomain_2018_10.csv")
# files_full_domain.append("FullDomain_2018_11.csv")
# files_full_domain.append("FullDomain_2018_12.csv")
files_surface = ["Surface_Black_Hills_2018.csv", "Surface_Wheatfield_2018.csv"]

surface_input_path = os.path.join('datasets', 'US_Brian', 'localformat', 'raw')
full_domain_input_path = os.path.join('C://', 'huge_datasets', 'FullDomain')
input_paths = [surface_input_path, full_domain_input_path]

output_path = os.path.join('datasets', 'US_Brian', 'localformat', 'cleaned')
translate_names_path = os.path.join('datasets', 'US_Brian', 'localformat', 'code_to_parameter_min.json')
for index, input_path in enumerate(input_paths):
    for data_file_name in os.listdir(input_path):
        translate_names = json.loads(open(translate_names_path, 'r').read())

        if data_file_name in files_full_domain:
            df = pd.read_csv(os.path.join(input_path, data_file_name), header=0)
            df = df.set_index(pd.to_datetime(df['time']))
            del df['time']
            del df.index.name
            df = df.rename(columns=translate_names)

            new_df = df.groupby(['x', 'y'])
            for leftf, rightf in new_df:
                if leftf == (-103.8269, 43.0815):
                    temp_df = pd.DataFrame()
                    temp_df['conc_tracer'] = rightf["ConcTracer"]
                    temp_df['dep_tracer'] = rightf["DepTracer"]
                    temp_df['datetime'] = rightf.index
                    temp_df.set_index('datetime', inplace=True)
                    del temp_df.index.name
                    df = temp_df
                    break

        elif data_file_name in files_surface:
            df = pd.read_csv(os.path.join(input_path, data_file_name), header=0)
            df = df.set_index([pd.to_datetime(dict(year=df['Year'], month=df['Month'], day=df['Day'], hour=df['Hour']))])
            del df['Year']
            del df['Month']
            del df['Day']
            del df['Hour']
            del df['Julian Day']
            df = df.rename(columns=translate_names)
            # print(df.head())
        else:
            print("ERROR: file {} not whitelisted. Please, check the whitelisted parameters".format(data_file_name))
            continue
        utils.write_dataframe_to_file(df, os.path.join(output_path, data_file_name))
        print(os.path.join(output_path, 'cleaned_' + data_file_name))
        # exit()
