# this guy will call 4-required-main_experiments_manager and simulate the future agent that will be implemented
import multiprocessing
import os

from models.nn_helper import main_nn_helper
from models.utils.utils import retrieve_file_with_info

run_prefix = 'kuwait-exp-CO'


def main():
    # must loop through experiments
    # then we call a modified version of 5-required....
    # where it just reads the arg/.. actually create a new function there!
    # and handle this script there using that specific function.
    full_run_path = os.path.join('experiments', run_prefix)
    for file in os.listdir(full_run_path):
        helper_params = retrieve_file_with_info(os.path.join(full_run_path, file))
        # print(helper_params)
        train_model(helper_params)


def train_model(helper_params):
    training_process = multiprocessing.Process(target=main_nn_helper, args=(helper_params,))
    training_process.start()
    training_process.join()


if __name__ == "__main__":
    main()
