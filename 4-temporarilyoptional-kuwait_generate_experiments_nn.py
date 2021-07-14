"""
File that generates configuration files for the NN creator

creates all possible permutations of given vectors -- each permutation is a config file
should be based on a internal config file.. a default file for this run.. where things will be static

"""
import hashlib
import itertools
import json
import os
import handle_data.utils as hd_utils

parameter_to_run = 'CO'

default_params = json.loads(open(os.path.join('Templates', 'default_nn_params.json'), 'r').read())
run_prefix = 'kuwait-exp-' + parameter_to_run

absolute_shared_folder = os.path.join('//winnas5', 'd$', 'ProjectData', 'ML_Forecast', 'KUWAIT')
refined_test_full_path = os.path.join(absolute_shared_folder, 'datasets', 'refined_datasets', 'final')
static_params = {
    # 'in_absolute_filename_dataset': os.path.join(refined_test_full_path, 'datasets', '840060190007.csv'),
    # 'in_absolute_filename_metadata': os.path.join(refined_test_full_path, 'metadata', '840060190007.json'),
    'out_absolute_path': os.path.join(absolute_shared_folder, 'models_runs'),
    'prefix_id': 'kuwait-' + parameter_to_run,
    # a str to be used as identifier for a specific dataset (a new folder to organize runs)
    'min_samples_to_train': 20000,  # minimum number of samples we should train the nn
    'lookback': 24,  # integer
    'timesteps': [x for x in range(1, 25)],  # vector of integers
    'batchsize': 5280,  # integer
    'epochs': 3000,  # integer
    'loss_functions': ['mse'],  # vector of strings with valid values
    'nn_metrics': ['mae', 'mse'],  # vector of strings with valid values. which metrics to evaluate
    # 'optimizer': 'nadam',  # string
    'train_split': 0.95,  # int the rest will go to test
    'validation_split': 0.052,  # float. 5% from training will go to validation
    'verbose_training': 0,  # int. depends on keras specification
    # 'output_vars': ['44201-1-refined'],  # vector of str. vars used on input and output
    # 'input_vars': ['44201-1-refined'],  # vector of str. vars only used on input
    'sampling_type': 'Hour',  # str
    'path_map_code_to_parameter': os.path.join(absolute_shared_folder, 'datasets', 'AQS_code_list',
                                               'code_to_parameter_min.json'),
    'path_map_code_to_unit': os.path.join(absolute_shared_folder, 'datasets', 'AQS_code_list', 'code_to_unit.json'),
    'force_cpu': False,  # bool
    'save_nn': True,  # bool
    'save_realpredict_test_data': True,  # bool
    'save_scaler': True,  # bool
    'save_model_graphs': True,  # bool
    'scaler': 'MinMaxScaler',  # str with valid values
    'callbacks': ['earlystopping'],  # vector of strings
    'nn_topology': None,  # not implemented yet
    'NN_type': 'LSTM',  # str
    'use_days_of_week': False,
    'use_hours_of_day': False,
    'num_buckets': False,
    'Comments': '',
}


def verify_keys_names(updated_dict):
    for key in updated_dict.keys():
        if key not in default_params.keys():
            print('Invalid key passed to update dict: ', key)
            print('Valid keys are:, ', default_params.keys())
            exit()


def update_static_dict(updated_dict):
    # this is actually a very important verification.. we assume that all the necessary paramters are defined before sending to lstm create
    verify_keys_names(updated_dict)
    static_params.update(updated_dict)


def is_static_dict_complete():
    for key in default_params.keys():
        if key not in static_params.keys():
            print('Key not passed to update dict: ', key)
            return False
    return True


def gen_possible_permutations(permute_with):
    keys = []
    values = []
    for key, value in permute_with.items():
        keys.append(key)
        values.append(value)

    return keys, list(itertools.product(*values))


def permute_parameters():
    # todo right now we need at least one to iterate and save.. we need to rethink this
    permute_with = {
        # 'epochs': [5000],
        # 'callbacks': [['earlystopping']],
        # 'batchsize': [5280],
        'optimizer': ['nadam']
    }

    # make sure permute_with keys does not exist on the static dict
    # for permute in permute_with.keys():
    # if permute in static_params.keys():
    #     print('Key present inside static parameters')
    #     print(permute)
    #     exit()

    # generates possible dict permutations
    keys, permutations = gen_possible_permutations(permute_with)

    permutations_dict = {}
    for permutation in permutations:
        for index, value in enumerate(permutation):
            permutations_dict.update({keys[index]: value})

        # here we update dict
        update_static_dict(permutations_dict)

        # compare dict keys with template dict keys
        if is_static_dict_complete():
            full_out_path = os.path.join('experiments', run_prefix, str(
                hashlib.sha1(json.dumps(static_params, sort_keys=True).encode()).hexdigest()) + '.json')
            # save json config
            hd_utils.write_json_to_file(static_params, full_out_path)
        else:
            print('Error, dict is not complete. Please provide all required keys.')
            exit()


def permute_files_and_vars():
    # here we iterate to create: in_absolute_filename_dataset, in_absolute_filename_metadata, input_vars, and output_vars
    # BUT NOW.. WE JUST WANT TO HAVE ALL FILES + PARAMETER.. AND REPEAT THE OTHER PARAMETERS
    # then.. you can permute parameters for all files.. or just for a specific file

    # give a parameter: say 44201
    # we have to go file by file (including metadata).. get eh available parameters.. build a dict with that path + identify
    # the poc for the parameter.. if there is more than one.. and/or if the pocc is different than 1..
    abs_datasets_path = os.path.join(refined_test_full_path, 'datasets')
    abs_metadata_path = os.path.join(refined_test_full_path, 'metadata')
    extra_input_parameter_code = []
    output_parameter_code = parameter_to_run
    for dataset in os.listdir(abs_datasets_path):
        supervised_prediction_var = []  # big loop here.. always repeat this guy on the inside loop when generating file
        multivariate_prediction_with = {x: [] for x in extra_input_parameter_code}  # inside loop here

        # if dataset.split('.')[0] != metadata.split('.')[0]:
        #     print('Dataset and metadata name are different!')
        #     exit()
        in_absolute_filename_dataset = os.path.join(abs_datasets_path, dataset)
        # in_absolute_filename_metadata = os.path.join(abs_metadata_path, metadata)

        df = hd_utils.pd.read_csv(in_absolute_filename_dataset, engine='python', skipfooter=0)
        df = df.set_index(df.columns[0])
        df.index.rename('id', inplace=True)

        for col in df.columns:
            col_param_code, col_type = col.rsplit('-', 1)
            for in_param_code in extra_input_parameter_code:
                if in_param_code == col_param_code and col_type == 'refined':
                    # add col to possible inputs vector (?)
                    multivariate_prediction_with[in_param_code].append(col)
            if output_parameter_code == col_param_code and col_type == 'refined':
                # add col to possible inputs vector (?)
                supervised_prediction_var.append(col)

        multivariate_prediction_with_permut = [item for key, item in multivariate_prediction_with.items()]

        for target_var in supervised_prediction_var:
            for permut in list(itertools.product(*multivariate_prediction_with_permut)):
                input_vars = [target_var, *permut]
                output_vars = [target_var]
                permuted_files = {
                    'in_absolute_filename_dataset': in_absolute_filename_dataset,
                    'in_absolute_filename_metadata': '',
                    'input_vars': input_vars,
                    'output_vars': output_vars
                }
                yield permuted_files
        #         print(permuted_files)
        # print(dataset, metadata)


def main():
    for permuted_files in permute_files_and_vars():
        update_static_dict(permuted_files)
        permute_parameters()
    return 0


if __name__ == "__main__":
    main()
