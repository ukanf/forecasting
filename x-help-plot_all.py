"""by Felipe Ukan - (c) 2019 Lakes Env. Software
adapted from: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

Simple script to just rapidly plot all columns from a csv file
This is very useful when we want to just check visualy if a generated csv is ok
"""
from pandas import read_csv
from matplotlib import pyplot
from handle_data.utils import utils
import os
absolute_shared_folder = os.path.join('//winnas5', 'd$', 'ProjectData', 'ML_Forecast', 'KUWAIT')
refined_test_full_path = os.path.join(absolute_shared_folder, 'datasets', 'final')
data_file_name = os.path.join(refined_test_full_path, 'newAMS17_AlRumaithiya_AMS_WRF_Merged.csv')
code_to_name_file = os.path.join(absolute_shared_folder, 'datasets', 'code_list', 'kuwait_code_to_unit.json')

dict_param = {
    # 'R_state_code': '06',
    # 'R_county_code': '029',
    'R_code_to_name_file': code_to_name_file,
    'PA_dividers_hours': [8760]
}

map_code_to_parameter, map_code_to_site, map_code_to_unit = utils.load_code_to_name_dicts(dict_param)

dataset = read_csv(data_file_name, header=0, index_col=0)
values = dataset.values
columns = dataset.columns

for group in range(len(dataset.columns)):
    pyplot.subplot(2, 1, 1)  # using 2 here just to plot faster..
    pyplot.grid(alpha=0.25)
    pyplot.plot(values[:, group])

    for num_divider in dict_param['PA_dividers_hours']:
        for divider in range(0, values[:, group].shape[0], num_divider):
            pyplot.axvline(x=divider, color="black", linewidth=1)

    pyplot.xlabel('Hour')
    try:
        pyplot.title(map_code_to_parameter[columns[group].split('_')[0]])
        pyplot.ylabel(map_code_to_unit[columns[group].split('_')[0]])
    except KeyError:
        pyplot.title(columns[group])
        pyplot.ylabel(columns[group])

    pyplot.show()
    pyplot.figure()
    # pyplot.savefig(str(i))
