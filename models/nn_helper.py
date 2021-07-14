"""
    We make sure the inputs are valid... create logger..
    Here we should find most of the documentation for the inputs too. This would be a buffer script that helps manage
    calling nn train
"""
import json
import os
import socket
import time
import logging
from models.nn_train import nn_multout_create
from models.utils.utils import setup_logger

default_params = json.loads(open(os.path.join('Templates', 'default_nn_params.json'), 'r').read())

def create_run_folder(unique_identifier, nn_params):
    hostname = socket.gethostname()
    path_unique_identifier = os.path.join(nn_params['out_absolute_path'], nn_params['prefix_id'], str(unique_identifier) + '-' + str(hostname))
    if os.path.exists(unique_identifier):
        print('ERROR: folder already exists')
        exit()
    else:
        # make all dirs we need
        os.makedirs(path_unique_identifier)
    return path_unique_identifier


def get_new_init_config(nn_params):
    time.sleep(2)  # to guarantee we get a different time
    unique_identifier = str(int(time.time()))
    logger_level = logging.DEBUG
    path_unique_identifier = create_run_folder(unique_identifier, nn_params)
    return unique_identifier, logger_level, path_unique_identifier


def verify_keys_names(updated_dict):
    for key in updated_dict.keys():
        if key not in default_params.keys():
            print('Invalid key passed to update dict: ', key)
            print('Valid keys are:, ', default_params.keys())
            exit()


def update_nn_params(nn_params, updated_nn_params):
    # this is actually a very important verification.. we assume that all the necessary paramters are defined before sending to lstm create
    verify_keys_names(updated_nn_params)
    nn_params.update(updated_nn_params)


# automating Neural Networks creation with a trier
def main_nn_helper(trier_params):
    """
    Unpacks every parameter into single parameters to give to different lstm_multout_create.. here we call individual create lstm methods with different parameters
    :param trier_params: dict {} with all the ranges for each parameter we want to try
    :return:
    """

    # create function: generate nn_params with default values if not given from trier_params
    # base_nn_params = create_default_nn_params()
    nn_params = default_params.copy()
    update_nn_params(nn_params, trier_params)

    unique_identifier, logger_level, path_unique_identifier = get_new_init_config(nn_params)
    setup_logger(unique_identifier, os.path.join(path_unique_identifier, unique_identifier + '.log'), logger_level)

    nn_multout_create(nn_params, unique_identifier, path_unique_identifier)

