import json
import logging
import os
import time
import pandas as pd


# def setup_logger(logger_name, log_file, level=logging.INFO):
#     """
#         Function to create a logger
#         :param logger_name: unique name to identify the logger
#         :param log_file: relative name with path of file of log
#         :param level: level of logging to be used
#     """
#     l = logging.getLogger(logger_name)
#     l.propagate = False
#     formatter = logging.Formatter('%(asctime)s %(name)-12s %(funcName)20s() %(levelname)-8s %(message)s')
#     fileHandler = logging.FileHandler(log_file, mode='w')
#     fileHandler.setFormatter(formatter)
#
#     l.setLevel(level)
#     l.addHandler(fileHandler)
import types


def write_dataframe_to_file(df, full_out_file_path):
    """ Saves the dataframe inside a new file in a new path
    by Felipe Ukan - 
    :param df:
    :param full_out_file_path:
    :return:
    """
    output_dir = os.path.dirname(full_out_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not file_exists(full_out_file_path):
        print('Saving file:', full_out_file_path)
        df.to_csv(full_out_file_path)
    else:
        # TODO MUST CONCATENATE
        existing_df = pd.read_csv(full_out_file_path, index_col=0)
        existing_df = pd.concat([existing_df, df], axis=1)
        existing_df.to_csv(full_out_file_path)
        print('File {} already exists. Concatenating content.', full_out_file_path)


class _PythonObjectEncoder(json.JSONEncoder):
    """ Hack used internally to unwrap 'set' to 'list' and timestamp to str.
    by Felipe Ukan - 
    """
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.__str__()
        return json.JSONEncoder.default(self, obj)


def write_json_to_file(info, full_out_file_path):
    """
    by Felipe Ukan - 
    :param info:
    :param full_out_file_path:
    :return:
    """
    output_dir = os.path.dirname(full_out_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not file_exists(full_out_file_path):
        print('Saving file:', full_out_file_path)
        with open(full_out_file_path, 'w') as outfile:
            json.dump(info, outfile, cls=_PythonObjectEncoder)
    else:
        with open(full_out_file_path, 'r+') as existing_file:
            existing_data = json.load(existing_file)
            existing_data.update(info)
            existing_file.seek(0)
            existing_file.truncate()
            json.dump(existing_data, existing_file, cls=_PythonObjectEncoder)
        print('File {} already exists. Updated with new data.', full_out_file_path)


def file_exists(saving_path):
    """ Check if the file already exists
    by Felipe Ukan - 
    :param saving_path:
    :return:
    """
    if os.path.isfile(saving_path):
        return True
    else:
        return False


def station_has_data(input_parameters, state, county, station):
    # print(download_params['C_prefix'])
    # print(state)
    # print(county)
    # print(station)
    prefix_in_path = os.path.join(input_parameters['C_prefix'], state, county)
    for year in input_parameters['years']:
        file_name = '840' + str(state) + str(county) + str(station)
        full_in_path = os.path.join(prefix_in_path, str(year) + '_' + str(year), 'frequency_1H')
        for file in os.listdir(full_in_path):
            if file[:12] == file_name:
                # print(year)
                return True

    return False

# def drop_other_cols(df, sites_to_keep):
#     """
#     by Felipe Ukan - 
#     :param df:
#     :param sites_to_keep:
#     :return:
#     """
#     for col in df.columns:
#         if col not in sites_to_keep:
#             del df[col]


def generate_input_parameters(input_parameters):
    """ Generates the possible inputs for cleaning and downloading data
    by Felipe Ukan - 
    :param input_parameters:
    :return:
    """
    try:
        for state in input_parameters:
            # print(state)
            # print(state['code'])
            for county in state['counties']:
                # print(county)
                # print(county['code'])
                for possible_parameters in county['parameters']:
                    # print(parameter)
                    possible_parameter = list(possible_parameters.items())
                    parameter = possible_parameter[0][0]

                    parameter_values = possible_parameter[0][1]
                    read_frequency = parameter_values['frequency']

                    for year in input_parameters['years']:
                        yield state['code'], county['code'], year, parameter, read_frequency
    except Exception as e:
        print('Aborting. Could not generate data. Error: {}. '.format(e))


def _generate_input_parameters_info(inner_refine_parameters):
    """
    by Felipe Ukan - 
    :param inner_refine_parameters:
    :return:
    """
    try:
        inner_read_frequency = inner_refine_parameters['R_frequency']
        inner_state = inner_refine_parameters['R_state_code']
        inner_county = inner_refine_parameters['R_county_code']

        for inner_year in inner_refine_parameters['R_year_range']:
            yield inner_state, inner_county, inner_year, inner_read_frequency

    except Exception as e:
        print('Aborting. Could not generate data. Error: {}. '.format(e))


def generate_input_parameters_refine(refine_parameters):
    """
    by Felipe Ukan - 
    :param refine_parameters:
    :return:
    """
    try:
        for item in _generate_input_parameters_info(refine_parameters):
            state, county, year, read_frequency = item

            for parameter in refine_parameters['R_selected_parameters']:

                if parameter not in refine_parameters['R_valid_parameters']:
                    print('Parameter: {} not valid.'.format(parameter))
                    print('Valid parameters are: {}'.format(refine_parameters['R_valid_parameters']))
                    continue

                yield state, county, year, read_frequency, parameter
    except Exception as e:
        print('Error generating input: {}'.format(e))

def get_info_refine(refine_parameters, root_dir):
    """
    :param refine_parameters:
    :param root_dir:
    :return:
    """
    for state in os.listdir(root_dir):
        for county in os.listdir(os.path.join(root_dir, state)):
            for year_year in os.listdir(os.path.join(root_dir, state, county)):
                final_dir = os.path.join(root_dir, state, county, year_year, 'frequency_1H')
                if os.path.isdir(final_dir):
                    for uid in os.listdir(final_dir):
                        if uid.split('_')[1] == 'meta.json':
                            full_path = os.path.join(final_dir, uid)
                            yield year_year, uid, full_path


def is_dataframe_ok(df):
    """ Checks if dataframe exists and if data inside is ok
    by Felipe Ukan - 
    :param df:
    :return:
    """
    # todo possibly add more verifications here.. more feedback
    if df.empty:
        return False
    return True


def _create_run_folder(unique_identifier, prefix_id):
    """ Creates a folder that will hold the info for the dataset being created
    by Felipe Ukan - 
    :param unique_identifier:
    :param prefix_id:
    :return:
    """
    path_unique_identifier = os.path.join(prefix_id, str(unique_identifier))
    if os.path.exists(unique_identifier):
        print('ERROR: folder already exists')
        exit()
    else:
        # make all dirs we need
        os.makedirs(path_unique_identifier)
    return path_unique_identifier


def get_new_init_config(refine_parameters):
    """ Creates new folder and unique number for dataset being created
    by Felipe Ukan - 
    :param refine_parameters:
    :return:
    """
    time.sleep(2)  # to guarantee we get a different time
    unique_identifier = str(int(time.time()))
    path_unique_identifier = _create_run_folder(unique_identifier, refine_parameters['R_prefix'])
    return unique_identifier, path_unique_identifier


def load_code_to_name_dicts(dict_params):
    """ Loads json file.. from state -> county (codes) get dictionaries to translate code to names
    by Felipe Ukan - 
    :param dict_params:
    :return:
    """
    try:
        with open(dict_params['R_code_to_name_min_file']) as file:
            data = json.load(file)
        return data
    except Exception as e:
        print('Error while loading code to real names dicts: {}'.format(e))
        return {}, {}, {}


def setup_logger(logger_name, log_file, level=logging.INFO):
    """ Function that creates a logger
    by Felipe Ukan - 
    :param logger_name: unique name to identify the logger
    :param log_file: relative name with path of file of log
    :param level: level of logging to be used
    """
    l = logging.getLogger(logger_name)
    l.propagate = False
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(funcName)20s() %(levelname)-8s %(message)s')
    if not os.path.isdir(os.path.split(log_file)[0]):
        os.makedirs(os.path.split(log_file)[0])
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

def timeit(method):
    """ Decorator that measures time of a function
    original source: https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
    by Felipe Ukan - 
    :param method:
    :return:
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r %2.2f sec' % (method.__name__, te - ts))
        return result

    return timed
