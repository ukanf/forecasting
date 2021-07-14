"""
File that creates the neural networks
"""
import os
from models.nn_helper import main_nn_helper


def run_bash(cmd):
    p = os.subprocess.Popen(cmd, shell=True, stdout=os.subprocess.PIPE, executable='/bin/bash')
    # out = p.stdout.read().strip()
    return p  # This is the stdout from the shell command


def kuwait_main():
    # *** Implementing ***
    # CALLS helper WITH PARAMETERS WE WANT (overwrites default params)
    absolute_shared_folder = os.path.join('//winnas5', 'd$', 'ProjectData', 'ML_Forecast', 'KUWAIT')
    refined_test_full_path = os.path.join(absolute_shared_folder, 'datasets', 'final')
    helper_params = {
        'in_absolute_filename_dataset': os.path.join(refined_test_full_path,
                                                     'newAMS11_AlAhmadi1_AMS_WRF_Merged.csv'),
        # 'in_absolute_filename_metadata': os.path.join(refined_test_full_path, 'metadata', '840060190007.json'),
        'out_absolute_path': os.path.join(absolute_shared_folder, 'models_runs'),
        'prefix_id': 'AlAhmadi1',
        'path_map_code_to_parameter': os.path.join(absolute_shared_folder, 'datasets', 'code_list',
                                                   'code_to_parameter_min.json'),
        'path_map_code_to_unit': os.path.join(absolute_shared_folder, 'datasets', 'code_list',
                                              'kuwait_code_to_unit.json'),
        'input_vars': ['O3'],  # only used as input
        'output_vars': ['O3'],  # only used as output
        'sampling_type': 'Hour',
        'epochs': 30,
        'min_samples_to_train': 10000,
        'timesteps': [x for x in range(1, 25)],
        # vector of ints: each n represent the timestep we want to predict (non-inclusive)
        'lookback': 24,
        'force_cpu': True,
        'loss_functions': ['mse'],
        'nn_metrics': ['mae', 'mse'],
        'scaler': False,
        'batchsize': 512,
        'NN_type': 'LSTM',
        'callbacks': ['earlystopping'],
        'use_days_of_week': False,
        'use_hours_of_day': False,
        'verbose_training': 1,
        'num_buckets': False,
        'Comments': ''
    }

    main_nn_helper(helper_params)


def main():
    # *** Implementing ***
    # CALLS helper WITH PARAMETERS WE WANT (overwrites default params)
    absolute_shared_folder = os.path.join('datasets', 'US_Brian', 'localformat')
    helper_params = {
        'in_absolute_filename_dataset': os.path.join('final_merged_reindexed.csv'),
        # 'in_absolute_filename_metadata': os.path.join(refined_test_full_path, 'metadata', '840060190007.json'),
        'out_absolute_path': os.path.join(absolute_shared_folder, 'models_runs'),
        'prefix_id': 'usbrian-tests',
        'path_map_code_to_parameter': os.path.join(absolute_shared_folder, 'datasets', 'code_list',
                                                   'code_to_parameter_min.json'),
        'path_map_code_to_unit': os.path.join(absolute_shared_folder, 'us_brian_code_to_unit.json'),
        'input_vars': ["conc_tracer_-104.9117,42.07426","dep_tracer_-104.9117,42.07426","conc_tracer_-104.9117,42.07426_rolling4back","dep_tracer_-104.9117,42.07426_rolling4back","conc_tracer_-104.948,42.05625","dep_tracer_-104.948,42.05625","conc_tracer_-104.948,42.05625_rolling4back","dep_tracer_-104.948,42.05625_rolling4back","conc_tracer_-104.9661,42.11029","dep_tracer_-104.9661,42.11029","conc_tracer_-104.9661,42.11029_rolling4back","dep_tracer_-104.9661,42.11029_rolling4back","conc_tracer_-104.9782,42.04725","dep_tracer_-104.9782,42.04725","conc_tracer_-104.9782,42.04725_rolling4back","dep_tracer_-104.9782,42.04725_rolling4back","conc_tracer_-105.0265,42.00221","dep_tracer_-105.0265,42.00221","conc_tracer_-105.0265,42.00221_rolling4back","dep_tracer_-105.0265,42.00221_rolling4back","conc_tracer_-105.0567,42.10128","dep_tracer_-105.0567,42.10128","conc_tracer_-105.0567,42.10128_rolling4back","dep_tracer_-105.0567,42.10128_rolling4back","wspd_P1_I-1J-1","tair_P1_I-1J-1","wdir_sin_P1_I-1J-1","wdir_cos_P1_I-1J-1","wspd_P1_I-1J-2","tair_P1_I-1J-2","wdir_sin_P1_I-1J-2","wdir_cos_P1_I-1J-2","wspd_P1_I-1J-3","tair_P1_I-1J-3","wdir_sin_P1_I-1J-3","wdir_cos_P1_I-1J-3","wspd_P1_I-1J-4","tair_P1_I-1J-4","wdir_sin_P1_I-1J-4","wdir_cos_P1_I-1J-4","wspd_P1_I-1J-5","tair_P1_I-1J-5","wdir_sin_P1_I-1J-5","wdir_cos_P1_I-1J-5","wspd_P1_I-1J0","tair_P1_I-1J0","wdir_sin_P1_I-1J0","wdir_cos_P1_I-1J0","wspd_P1_I-1J1","tair_P1_I-1J1","wdir_sin_P1_I-1J1","wdir_cos_P1_I-1J1","wspd_P1_I-1J2","tair_P1_I-1J2","wdir_sin_P1_I-1J2","wdir_cos_P1_I-1J2","wspd_P1_I-1J3","tair_P1_I-1J3","wdir_sin_P1_I-1J3","wdir_cos_P1_I-1J3","wspd_P1_I-1J4","tair_P1_I-1J4","wdir_sin_P1_I-1J4","wdir_cos_P1_I-1J4","wspd_P1_I-1J5","tair_P1_I-1J5","wdir_sin_P1_I-1J5","wdir_cos_P1_I-1J5","wspd_P1_I-2J-1","tair_P1_I-2J-1","wdir_sin_P1_I-2J-1","wdir_cos_P1_I-2J-1","wspd_P1_I-2J-2","tair_P1_I-2J-2","wdir_sin_P1_I-2J-2","wdir_cos_P1_I-2J-2","wspd_P1_I-2J-3","tair_P1_I-2J-3","wdir_sin_P1_I-2J-3","wdir_cos_P1_I-2J-3","wspd_P1_I-2J-4","tair_P1_I-2J-4","wdir_sin_P1_I-2J-4","wdir_cos_P1_I-2J-4","wspd_P1_I-2J-5","tair_P1_I-2J-5","wdir_sin_P1_I-2J-5","wdir_cos_P1_I-2J-5","wspd_P1_I-2J0","tair_P1_I-2J0","wdir_sin_P1_I-2J0","wdir_cos_P1_I-2J0","wspd_P1_I-2J1","tair_P1_I-2J1","wdir_sin_P1_I-2J1","wdir_cos_P1_I-2J1","wspd_P1_I-2J2","tair_P1_I-2J2","wdir_sin_P1_I-2J2","wdir_cos_P1_I-2J2","wspd_P1_I-2J3","tair_P1_I-2J3","wdir_sin_P1_I-2J3","wdir_cos_P1_I-2J3","wspd_P1_I-2J4","tair_P1_I-2J4","wdir_sin_P1_I-2J4","wdir_cos_P1_I-2J4","wspd_P1_I-2J5","tair_P1_I-2J5","wdir_sin_P1_I-2J5","wdir_cos_P1_I-2J5","wspd_P1_I-3J-1","tair_P1_I-3J-1","wdir_sin_P1_I-3J-1","wdir_cos_P1_I-3J-1","wspd_P1_I-3J-2","tair_P1_I-3J-2","wdir_sin_P1_I-3J-2","wdir_cos_P1_I-3J-2","wspd_P1_I-3J-3","tair_P1_I-3J-3","wdir_sin_P1_I-3J-3","wdir_cos_P1_I-3J-3","wspd_P1_I-3J-4","tair_P1_I-3J-4","wdir_sin_P1_I-3J-4","wdir_cos_P1_I-3J-4","wspd_P1_I-3J-5","tair_P1_I-3J-5","wdir_sin_P1_I-3J-5","wdir_cos_P1_I-3J-5","wspd_P1_I-3J0","tair_P1_I-3J0","wdir_sin_P1_I-3J0","wdir_cos_P1_I-3J0","wspd_P1_I-3J1","tair_P1_I-3J1","wdir_sin_P1_I-3J1","wdir_cos_P1_I-3J1","wspd_P1_I-3J2","tair_P1_I-3J2","wdir_sin_P1_I-3J2","wdir_cos_P1_I-3J2","wspd_P1_I-3J3","tair_P1_I-3J3","wdir_sin_P1_I-3J3","wdir_cos_P1_I-3J3","wspd_P1_I-3J4","tair_P1_I-3J4","wdir_sin_P1_I-3J4","wdir_cos_P1_I-3J4","wspd_P1_I-3J5","tair_P1_I-3J5","wdir_sin_P1_I-3J5","wdir_cos_P1_I-3J5","wspd_P1_I-4J-1","tair_P1_I-4J-1","wdir_sin_P1_I-4J-1","wdir_cos_P1_I-4J-1","wspd_P1_I-4J-2","tair_P1_I-4J-2","wdir_sin_P1_I-4J-2","wdir_cos_P1_I-4J-2","wspd_P1_I-4J-3","tair_P1_I-4J-3","wdir_sin_P1_I-4J-3","wdir_cos_P1_I-4J-3","wspd_P1_I-4J-4","tair_P1_I-4J-4","wdir_sin_P1_I-4J-4","wdir_cos_P1_I-4J-4","wspd_P1_I-4J-5","tair_P1_I-4J-5","wdir_sin_P1_I-4J-5","wdir_cos_P1_I-4J-5","wspd_P1_I-4J0","tair_P1_I-4J0","wdir_sin_P1_I-4J0","wdir_cos_P1_I-4J0","wspd_P1_I-4J1","tair_P1_I-4J1","wdir_sin_P1_I-4J1","wdir_cos_P1_I-4J1","wspd_P1_I-4J2","tair_P1_I-4J2","wdir_sin_P1_I-4J2","wdir_cos_P1_I-4J2","wspd_P1_I-4J3","tair_P1_I-4J3","wdir_sin_P1_I-4J3","wdir_cos_P1_I-4J3","wspd_P1_I-4J4","tair_P1_I-4J4","wdir_sin_P1_I-4J4","wdir_cos_P1_I-4J4","wspd_P1_I-4J5","tair_P1_I-4J5","wdir_sin_P1_I-4J5","wdir_cos_P1_I-4J5","wspd_P1_I-5J-1","tair_P1_I-5J-1","wdir_sin_P1_I-5J-1","wdir_cos_P1_I-5J-1","wspd_P1_I-5J-2","tair_P1_I-5J-2","wdir_sin_P1_I-5J-2","wdir_cos_P1_I-5J-2","wspd_P1_I-5J-3","tair_P1_I-5J-3","wdir_sin_P1_I-5J-3","wdir_cos_P1_I-5J-3","wspd_P1_I-5J-4","tair_P1_I-5J-4","wdir_sin_P1_I-5J-4","wdir_cos_P1_I-5J-4","wspd_P1_I-5J-5","tair_P1_I-5J-5","wdir_sin_P1_I-5J-5","wdir_cos_P1_I-5J-5","wspd_P1_I-5J0","tair_P1_I-5J0","wdir_sin_P1_I-5J0","wdir_cos_P1_I-5J0","wspd_P1_I-5J1","tair_P1_I-5J1","wdir_sin_P1_I-5J1","wdir_cos_P1_I-5J1","wspd_P1_I-5J2","tair_P1_I-5J2","wdir_sin_P1_I-5J2","wdir_cos_P1_I-5J2","wspd_P1_I-5J3","tair_P1_I-5J3","wdir_sin_P1_I-5J3","wdir_cos_P1_I-5J3","wspd_P1_I-5J4","tair_P1_I-5J4","wdir_sin_P1_I-5J4","wdir_cos_P1_I-5J4","wspd_P1_I-5J5","tair_P1_I-5J5","wdir_sin_P1_I-5J5","wdir_cos_P1_I-5J5","wspd_P1_I0J-1","tair_P1_I0J-1","wdir_sin_P1_I0J-1","wdir_cos_P1_I0J-1","wspd_P1_I0J-2","tair_P1_I0J-2","wdir_sin_P1_I0J-2","wdir_cos_P1_I0J-2","wspd_P1_I0J-3","tair_P1_I0J-3","wdir_sin_P1_I0J-3","wdir_cos_P1_I0J-3","wspd_P1_I0J-4","tair_P1_I0J-4","wdir_sin_P1_I0J-4","wdir_cos_P1_I0J-4","wspd_P1_I0J-5","tair_P1_I0J-5","wdir_sin_P1_I0J-5","wdir_cos_P1_I0J-5","wspd_P1_I0J0","tair_P1_I0J0","wdir_sin_P1_I0J0","wdir_cos_P1_I0J0","wspd_P1_I0J1","tair_P1_I0J1","wdir_sin_P1_I0J1","wdir_cos_P1_I0J1","wspd_P1_I0J2","tair_P1_I0J2","wdir_sin_P1_I0J2","wdir_cos_P1_I0J2","wspd_P1_I0J3","tair_P1_I0J3","wdir_sin_P1_I0J3","wdir_cos_P1_I0J3","wspd_P1_I0J4","tair_P1_I0J4","wdir_sin_P1_I0J4","wdir_cos_P1_I0J4","wspd_P1_I0J5","tair_P1_I0J5","wdir_sin_P1_I0J5","wdir_cos_P1_I0J5","wspd_P1_I1J-1","tair_P1_I1J-1","wdir_sin_P1_I1J-1","wdir_cos_P1_I1J-1","wspd_P1_I1J-2","tair_P1_I1J-2","wdir_sin_P1_I1J-2","wdir_cos_P1_I1J-2","wspd_P1_I1J-3","tair_P1_I1J-3","wdir_sin_P1_I1J-3","wdir_cos_P1_I1J-3","wspd_P1_I1J-4","tair_P1_I1J-4","wdir_sin_P1_I1J-4","wdir_cos_P1_I1J-4","wspd_P1_I1J-5","tair_P1_I1J-5","wdir_sin_P1_I1J-5","wdir_cos_P1_I1J-5","wspd_P1_I1J0","tair_P1_I1J0","wdir_sin_P1_I1J0","wdir_cos_P1_I1J0","wspd_P1_I1J1","tair_P1_I1J1","wdir_sin_P1_I1J1","wdir_cos_P1_I1J1","wspd_P1_I1J2","tair_P1_I1J2","wdir_sin_P1_I1J2","wdir_cos_P1_I1J2","wspd_P1_I1J3","tair_P1_I1J3","wdir_sin_P1_I1J3","wdir_cos_P1_I1J3","wspd_P1_I1J4","tair_P1_I1J4","wdir_sin_P1_I1J4","wdir_cos_P1_I1J4","wspd_P1_I1J5","tair_P1_I1J5","wdir_sin_P1_I1J5","wdir_cos_P1_I1J5","wspd_P1_I2J-1","tair_P1_I2J-1","wdir_sin_P1_I2J-1","wdir_cos_P1_I2J-1","wspd_P1_I2J-2","tair_P1_I2J-2","wdir_sin_P1_I2J-2","wdir_cos_P1_I2J-2","wspd_P1_I2J-3","tair_P1_I2J-3","wdir_sin_P1_I2J-3","wdir_cos_P1_I2J-3","wspd_P1_I2J-4","tair_P1_I2J-4","wdir_sin_P1_I2J-4","wdir_cos_P1_I2J-4","wspd_P1_I2J-5","tair_P1_I2J-5","wdir_sin_P1_I2J-5","wdir_cos_P1_I2J-5","wspd_P1_I2J0","tair_P1_I2J0","wdir_sin_P1_I2J0","wdir_cos_P1_I2J0","wspd_P1_I2J1","tair_P1_I2J1","wdir_sin_P1_I2J1","wdir_cos_P1_I2J1","wspd_P1_I2J2","tair_P1_I2J2","wdir_sin_P1_I2J2","wdir_cos_P1_I2J2","wspd_P1_I2J3","tair_P1_I2J3","wdir_sin_P1_I2J3","wdir_cos_P1_I2J3","wspd_P1_I2J4","tair_P1_I2J4","wdir_sin_P1_I2J4","wdir_cos_P1_I2J4","wspd_P1_I2J5","tair_P1_I2J5","wdir_sin_P1_I2J5","wdir_cos_P1_I2J5","wspd_P1_I3J-1","tair_P1_I3J-1","wdir_sin_P1_I3J-1","wdir_cos_P1_I3J-1","wspd_P1_I3J-2","tair_P1_I3J-2","wdir_sin_P1_I3J-2","wdir_cos_P1_I3J-2","wspd_P1_I3J-3","tair_P1_I3J-3","wdir_sin_P1_I3J-3","wdir_cos_P1_I3J-3","wspd_P1_I3J-4","tair_P1_I3J-4","wdir_sin_P1_I3J-4","wdir_cos_P1_I3J-4","wspd_P1_I3J-5","tair_P1_I3J-5","wdir_sin_P1_I3J-5","wdir_cos_P1_I3J-5","wspd_P1_I3J0","tair_P1_I3J0","wdir_sin_P1_I3J0","wdir_cos_P1_I3J0","wspd_P1_I3J1","tair_P1_I3J1","wdir_sin_P1_I3J1","wdir_cos_P1_I3J1","wspd_P1_I3J2","tair_P1_I3J2","wdir_sin_P1_I3J2","wdir_cos_P1_I3J2","wspd_P1_I3J3","tair_P1_I3J3","wdir_sin_P1_I3J3","wdir_cos_P1_I3J3","wspd_P1_I3J4","tair_P1_I3J4","wdir_sin_P1_I3J4","wdir_cos_P1_I3J4","wspd_P1_I3J5","tair_P1_I3J5","wdir_sin_P1_I3J5","wdir_cos_P1_I3J5","wspd_P1_I4J-1","tair_P1_I4J-1","wdir_sin_P1_I4J-1","wdir_cos_P1_I4J-1","wspd_P1_I4J-2","tair_P1_I4J-2","wdir_sin_P1_I4J-2","wdir_cos_P1_I4J-2","wspd_P1_I4J-3","tair_P1_I4J-3","wdir_sin_P1_I4J-3","wdir_cos_P1_I4J-3","wspd_P1_I4J-4","tair_P1_I4J-4","wdir_sin_P1_I4J-4","wdir_cos_P1_I4J-4","wspd_P1_I4J-5","tair_P1_I4J-5","wdir_sin_P1_I4J-5","wdir_cos_P1_I4J-5","wspd_P1_I4J0","tair_P1_I4J0","wdir_sin_P1_I4J0","wdir_cos_P1_I4J0","wspd_P1_I4J1","tair_P1_I4J1","wdir_sin_P1_I4J1","wdir_cos_P1_I4J1","wspd_P1_I4J2","tair_P1_I4J2","wdir_sin_P1_I4J2","wdir_cos_P1_I4J2","wspd_P1_I4J3","tair_P1_I4J3","wdir_sin_P1_I4J3","wdir_cos_P1_I4J3","wspd_P1_I4J4","tair_P1_I4J4","wdir_sin_P1_I4J4","wdir_cos_P1_I4J4","wspd_P1_I4J5","tair_P1_I4J5","wdir_sin_P1_I4J5","wdir_cos_P1_I4J5","wspd_P1_I5J-1","tair_P1_I5J-1","wdir_sin_P1_I5J-1","wdir_cos_P1_I5J-1","wspd_P1_I5J-2","tair_P1_I5J-2","wdir_sin_P1_I5J-2","wdir_cos_P1_I5J-2","wspd_P1_I5J-3","tair_P1_I5J-3","wdir_sin_P1_I5J-3","wdir_cos_P1_I5J-3","wspd_P1_I5J-4","tair_P1_I5J-4","wdir_sin_P1_I5J-4","wdir_cos_P1_I5J-4","wspd_P1_I5J-5","tair_P1_I5J-5","wdir_sin_P1_I5J-5","wdir_cos_P1_I5J-5","wspd_P1_I5J0","tair_P1_I5J0","wdir_sin_P1_I5J0","wdir_cos_P1_I5J0","wspd_P1_I5J1","tair_P1_I5J1","wdir_sin_P1_I5J1","wdir_cos_P1_I5J1","wspd_P1_I5J2","tair_P1_I5J2","wdir_sin_P1_I5J2","wdir_cos_P1_I5J2","wspd_P1_I5J3","tair_P1_I5J3","wdir_sin_P1_I5J3","wdir_cos_P1_I5J3","wspd_P1_I5J4","tair_P1_I5J4","wdir_sin_P1_I5J4","wdir_cos_P1_I5J4","wspd_P1_I5J5","tair_P1_I5J5","wdir_sin_P1_I5J5","wdir_cos_P1_I5J5"],  # only used as input
        'output_vars': ["conc_tracer_-104.9117,42.07426_rolling4back","dep_tracer_-104.9117,42.07426_rolling4back","conc_tracer_-104.948,42.05625_rolling4back","dep_tracer_-104.948,42.05625_rolling4back","conc_tracer_-104.9661,42.11029_rolling4back","dep_tracer_-104.9661,42.11029_rolling4back","conc_tracer_-104.9782,42.04725_rolling4back","dep_tracer_-104.9782,42.04725_rolling4back","conc_tracer_-105.0265,42.00221_rolling4back","dep_tracer_-105.0265,42.00221_rolling4back","conc_tracer_-105.0567,42.10128_rolling4back","dep_tracer_-105.0567,42.10128_rolling4back"],  # only used as output
        'sampling_type': 'Hour',
        'epochs': 45,
        "min_samples_to_train": 5000,
        'timesteps': [x for x in range(1, 4)],
        # vector of ints: each n represent the timestep we want to predict (non-inclusive)
        'lookback': 4,
        'force_cpu': True,
        'loss_functions': ['mse'],
        'nn_metrics': ['mae', 'mse', 'accuracy'],
        'scaler': 'MinMaxScaler',
        'batchsize': 256,
        'NN_type': 'LSTM',
        # 'callbacks': ['earlystopping'],
        'use_days_of_week': False,
        'use_hours_of_day': True,
        'verbose_training': 0,
        'num_buckets': False,
        'Comments': ''
    }

    main_nn_helper(helper_params)


if __name__ == "__main__":
    kuwait_main()
