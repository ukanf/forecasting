"""
Very useful script to compare two or more datasets!
1. This is good to visualize original and refined data.
2. This is also useful to plot more than one parameter.
"""
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# read files that has data
# more than 1 file
# get all the predict data into 1 2d array
# x axes is the length of each array (they must have the same size)

# check size of the two
# check index
datasets = {}
datasets_paths = []
datasets_labels = []
type = 'testPredict'
num_steps_graph = 24
current_ts = ['t+1', 't+2', 't+3', 't+4', 't+5', 't+6']
location = '-104.9782,42.04725'

folder_path = os.path.join('datasets', 'US_Brian', 'localformat', 'models_runs', 'usbrian-tests', '1573231527-PLUTO')

for current_t in current_ts:
    external_datasets_paths = [os.path.join(folder_path, 'realpredict-conc_tracer_'+location+'_rolling4back_' + current_t + '.csv'), os.path.join(folder_path, 'realpredict-conc_tracer_'+location+'_rolling4back_' + current_t + '.csv')]
    external_labels = ['testY', 'testPredict']
    external_labels_translate = {
        'testY': 'Measured',
        'testPredict': 'Predicted'
    }
    start_test_data_in_external = 75
    div_num = 3
    # graph_upper_limit = 0.15
    # graph_lower_limit = -0.01

    if external_datasets_paths:
        for external_dataset_filename, external_label in zip(external_datasets_paths, external_labels):
            df = pd.read_csv(external_dataset_filename, header=0, index_col=0)
            datasets[external_label] = df[external_label].values[start_test_data_in_external:start_test_data_in_external+num_steps_graph]

        for external_label in external_labels:
            plt.plot(datasets[external_label], label=external_labels_translate[external_label])

    # plt.axvline(x=0, color='green', linestyle=':', label='24 hours split')
    # for xc in range(24, num_steps_graph, 24):
    #     plt.axvline(x=xc, color='green', linestyle=':')

    # plt.title(str(datasets_labels) + str(external_labels))
    plt.title('Concentration 24h ' + current_t)
    plt.xlabel('Hour')
    plt.ylabel('log(x*10$^{20}$)')
    # plt.show()
    plt.legend()
    new_label_external_div = str(external_labels) + 'DIV' + str(div_num) + '_'
    plt.grid()
    # plt.gca().set_ylim([graph_lower_limit, graph_upper_limit])
    # plt.show()
    # plt.savefig(str(datasets_labels) + str(new_label_external_div) + str(num_steps_graph) + str(type) + '.png')

    # plt.plot(testY, label='real_data')
    # plt.plot(testPredict, label='prediction')
    # plt.title(str(var_name))
    # plt.xlabel('epochs')
    # plt.ylabel('concentration')
    # plt.legend()
    savefile_name = os.path.join(folder_path, 'loc' + location + 'nsteps' + str(num_steps_graph) +
                                 '_currentt' + current_t + '.png')
    plt.savefig(savefile_name)
    plt.clf()
    plt.cla()
    plt.close()
    # plt.show()