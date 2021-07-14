"""
LSTM RNN for predicting timeseries
"""
import itertools

from keras.callbacks import EarlyStopping

__author__ = "Felipe Ukan Pereira"
__copyright__ = "(c) 2019 Lakes Environmental Software Inc. Felipe Ukan Pereira"
__license__ = "Proprietary"

import numpy as np
import logging
import os
from argparse import Namespace
import tensorflow as tf

# MANUAL FORCE CPU FOR NOW
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, CuDNNLSTM, CuDNNGRU, GRU, Conv1D, MaxPooling1D, Flatten
import threading
import time
from sys import exit
import models.utils as utils


# np.set_printoptions(threshold=np.nan)


# epochs, look_back, predict_vars, time_steps, filename, optimizer='nadam',testtrainlossgraph=False, batch_size=512, loss_function='mse', train_split=0.8, force_cpu=True, output_save_path=False

@utils.timeit
def nn_multout_create(nn_params, unique_identifier, path_unique_identifier):
    """
    Main flow for creating the NN
    by Felipe Ukan - (c) 2019 Lakes Env. Software
    :param nn_params: dictionary with parameters to be used for creating the NN and handling everything else,
    like saving graphs, using tensorboard, etc.
    :param unique_identifier: a str with the unique number for this run
    :param path_unique_identifier: a str as path created beforehand so we know where to save everything
    """
    if not nn_params or not unique_identifier:
        print('Error, wrong parameters.. no unique identifier or parameters')
        exit()

    logger = logging.getLogger(unique_identifier)
    logger.debug('Starting Multout')

    # fix random seed for reproducibility
    np.random.seed(7)

    # open nn_param dictionary
    v = Namespace(**nn_params)

    # saves configuration file as json
    utils.write_json_to_file(nn_params, os.path.join(path_unique_identifier, 'config.json'))

    # creates a local copy of the given params
    input_dict = nn_params.copy()

    # initializes the keras callback vector
    keras_callbacks = utils.resolve_keras_callbacks(v, unique_identifier, path_unique_identifier)

    # reads csv file and sets index column
    df = utils.read_csvdata(v, unique_identifier, skipfooter=0)
    df.dropna(how='all', inplace=True)

    # verification that timestep is in range
    for time_step in v.timesteps:
        if 1 > time_step or time_step > 48:
            logger.error('Cannot continue.. time step must be between 1 and 48')
            logger.critical('Timestep error..')
            exit()
    # conversion just to make the input more readable to the user.. user will give 1, 2, ..., 24 timesteps but we just convert here. (so we can get the element 0 from array if necessary by professor forcing and scheduling sample)
    v.timesteps = [x - 1 for x in v.timesteps]

    # create a vector of axisX and Y for each param, where axisY is the target variables vectors
    # here happens most of the preprocessing.. we create new features.. remove nans, trim dataset, select inputs and outputs, shift columns for different timestep predictions, and we have buckets too (discretizes inputs)
    axisX, axisY, input_vars_names, output_vars_names = utils.create_XY_arrays_multout(df, unique_identifier, v)
    swapedY = axisY.swapaxes(0, 1)

    # saves the min and max values for each component
    minY_values = [min(_) for _ in swapedY]
    maxY_values = [max(_) for _ in swapedY]

    # applies scaler if required.. saves it.. and returns the Y so we can revert predictions back later
    axisX, axisY, scalerY = utils.resolve_scaler(axisX, axisY, input_vars_names, output_vars_names, unique_identifier,
                                                 path_unique_identifier, v)

    # prepare output arrays
    trainX, testX, trainY, testY = utils.prepare_XY_arrays(axisX, axisY, unique_identifier, v)


    # print(trainX)

    # exit()

    # updating metrics dictionary with new data
    input_dict.update({
        'Train split %': int(round(v.train_split * 100)),
        'Validation split % (percentage from train split to use during training for validation)': int(
            round(v.validation_split * 100)),
        'Test split %': int(round((1 - v.train_split) * 100)),
        'minY_values': minY_values,
        'maxY_values': maxY_values,
        'trainX_shape (samples, timeseries/lookback, parameters)': trainX.shape,
        'testX_shape (samples, timeseries/lookback, parameters)': testX.shape,
        'trainY_shape (samples, outputs/cloumns to predict)': trainY.shape,
        'testY_shape (samples, outputs/cloumns to predict)': testY.shape,
    })

    # Network declaration
    model = resolve_network_declaration(trainX, trainY, unique_identifier, v)

    # compiles the model
    model.compile(loss=v.loss_functions, optimizer=v.optimizer, metrics=v.nn_metrics)

    # initializes thread that updates the logger
    threading.Thread(daemon=True, target=update_logger_and_info, args=(logger, model, input_dict,)).start()

    # fits the model
    history = model.fit(trainX, trainY, epochs=v.epochs, batch_size=v.batchsize,
                        validation_split=v.validation_split, shuffle=False,
                        verbose=v.verbose_training, callbacks=keras_callbacks)

    ######### Finished training now we start saving results
    # saves model to files
    if v.save_nn: utils.save_nn(model, path_unique_identifier, unique_identifier)

    # make predictions
    # trainPredict = model.predict(trainX)
    testPredict = model.predict(testX, batch_size=v.batchsize)

    # test loss and training loss graph. It can help understand the optimal epochs size and help check if the model
    # is overfitting or underfitting.
    if v.save_model_graphs: utils.create_modelloss_graph(history, os.path.join(path_unique_identifier, 'modelloss'))

    # prints model to screen.
    model.summary()

    # invert predictions
    if scalerY:
        testPredict = scalerY.inverse_transform(testPredict)
        testY = scalerY.inverse_transform(testY)
    else: testY = np.squeeze(np.asarray(testY))
    # above need to convert to numpy array (it was a matrix) because if we dont enter the v.scaler it will throw an
    # error because they would be incompatible

    testPredict = testPredict.swapaxes(0, 1)  # column becomes line
    if len(v.timesteps) > 1: testY = testY.swapaxes(0, 1)  # column becomes line
    else: testY = [testY]
    # above result is a single vector here because we are predicting only 1
    # var. but we iterate over it, so it must be the first vector of a vector.

    # temporary array to allow us to iterate over all the results.
    temp_all_out_vars = [str(item[0]) + '_t+' + str(item[1] + 1) for item in
                         itertools.product(v.output_vars, v.timesteps)]
    # one loop for min_values, max_values, testY, testPredict...
    for var_name_timesteps, min_val, max_val, var_test, var_predict in zip(
            temp_all_out_vars, minY_values, maxY_values, testY, testPredict):

        logger.info(
            'Score for {0} - : {1:.5f} RMSE'.format(var_name_timesteps, utils.calculate_RMSE(var_test, var_predict)))

        var_name, var_timesteps = var_name_timesteps.split('_t+')

        try:
            parameter_code, poc, dt_type = var_name.split('-')
            parameter_name = str(utils.retrieve_file_with_info(v.path_map_code_to_parameter)[parameter_code])
        except Exception as e:
            # parameter_code, poc, dt_type = var_name, var_name, var_name
            parameter_name = var_name.split('-')[0]
            logger.warning('Error {}. When converting parameter code to parameter name.'.format(e))

        try:
            # unique_unit_code = int(float(utils.retrieve_unique_unit_code(v.in_absolute_filename_metadata, var_name)))
            unique_unit = utils.retrieve_file_with_info(v.path_map_code_to_unit)[str(var_name).zfill(3)]
            print(unique_unit)
        except Exception as e:
            unique_unit = "[µg/m³]"
            logger.warning('Error {}. When converting unit code to unit name.'.format(e))

        utils.create_realpredict_graph(var_test[0:300], var_predict[0:300],
                                       title=parameter_name + ' t+' + var_timesteps,
                                       savefile_path=os.path.join(path_unique_identifier,
                                                                  'realpredict-' + str(var_name_timesteps)),
                                       x_axis_label=v.sampling_type,
                                       y_axis_label=unique_unit)

        if v.save_realpredict_test_data:
            path_to_save = os.path.join(path_unique_identifier, 'realpredict-' + str(var_name_timesteps) + '.csv')
            utils.save_realpredict_data(var_test, var_predict, path_to_save)

    # resets current graph
    utils.keras.backend.clear_session()

    # closes logger
    logger.removeHandler(logger)

    return 0


def create_custom_nn(topology):
    pass


def resolve_network_declaration(trainX, trainY, unique_identifier, v):
    gru_valid_str = ['GRU', 'gru']
    cnn_valid_str = ['CNN', 'cnn']
    lstm_valid_str = ['LSTM', 'lstm']

    if v.force_cpu:  # means we are running only on CPU
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # https://github.com/keras-team/keras/issues/152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        if v.NN_type in gru_valid_str:
            return create_multout_gru_CPU(trainX, trainY, v.batchsize)
        elif v.NN_type in lstm_valid_str:
            print('Using LSTM CPU as default')
            return create_multout_lstm_CPU(trainX, trainY, v.batchsize)
        else:
            print('Invalid NN type')
            exit(1)
    else:  # means we will run the model on GPU
        if v.NN_type in gru_valid_str:
            return create_multout_gru_CUDA(trainX, trainY, v.batchsize)
        elif v.NN_type in cnn_valid_str:
            return create_multout_cnn(trainX, trainY, v.batchsize)
        elif v.NN_type in lstm_valid_str:
            print('Using LSTM GPU as default')
            return create_multout_lstm_CUDA(trainX, trainY, v.batchsize)
        else:
            print('Invalid NN type')
            exit(1)


def create_multout_gru_CUDA(trainX, trainY, batchsize, topology=None):
    if topology:
        t = Namespace(**topology)
        return create_custom_nn(t)
    else:
        model = Sequential()

        model.add(
            CuDNNGRU(22, return_sequences=True,
                     input_shape=(trainX.shape[1], trainX.shape[2])))

        model.add(Dropout(0.2))

        model.add(CuDNNGRU(16, return_sequences=True))

        model.add(CuDNNGRU(12, return_sequences=False))

        model.add(Dense(trainY.shape[1], activation='linear'))

        return model


def create_multout_lstm_CUDA(trainX, trainY, batchsize, topology=None):
    if topology:
        t = Namespace(**topology)
        return create_custom_nn(t)
    else:
        model = Sequential()

        model.add(
            LSTM(32, return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))

        print(trainX[0])

        exit()

        model.add(Dropout(0.6))

        model.add(LSTM(32, return_sequences=False))

        model.add(Dropout(0.4))

        model.add(Dense(64, activation='relu'))

        model.add(Dense(32, activation='relu'))

        model.add(Dropout(0.4))

        model.add(Dense(trainY.shape[1], activation='linear'))

        return model


def create_multout_lstm_CPU(trainX, trainY, batchsize, topology=None):
    if topology:
        t = Namespace(**topology)
        return create_custom_nn(t)
    else:
        model = Sequential()

        model.add(LSTM(24 + 12, activation="relu", return_sequences=True,
                      input_shape=(trainX.shape[1], trainX.shape[2])))  # , dropout=0.1, recurrent_dropout=0.1, ))

        model.add(LSTM(24, activation="relu", return_sequences=False))

        model.add(Dense(trainY.shape[1], activation='linear'))

        return model


def create_multout_gru_CPU(trainX, trainY, batchsize, topology=None):
    if topology:
        t = Namespace(**topology)
        return create_custom_nn(t)
    else:
        model = Sequential()

        model.add(
            GRU(22, unit_forget_bias=True,
                input_shape=(trainX.shape[1], trainX.shape[2])))

        model.add(Dropout(0.2))

        model.add(GRU(16, unit_forget_bias=True))

        model.add(GRU(12, unit_forget_bias=True))

        model.add(Dense(trainY.shape[1], activation='linear'))

        return model


def Dcreate_multout_lstm_CPU(trainX, trainY, batchsize, topology=None):
    if topology:
        t = Namespace(**topology)
        return create_custom_nn(t)
    else:
        model = Sequential()

        input_nodes = int(trainX.shape[2] * 2)

        model.add(
            LSTM(input_nodes, activation='sigmoid', recurrent_activation='tanh',
                 input_shape=(trainX.shape[1], trainX.shape[2])))

        # model.add(Dropout(0.2))

        # model.add(LSTM(16, unit_forget_bias=True, return_sequences=True, activation='relu'))

        # model.add(LSTM(12, unit_forget_bias=True, return_sequences=False, activation='relu'))

        model.add(Dense(trainY.shape[1], activation='relu'))

        return model


def create_multout_cnn(trainX, trainY, batchsize, topology=None):
    if topology:
        t = Namespace(**topology)
        return create_custom_nn(t)
    else:
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=7, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(trainY.shape[1]))

        return model


def update_logger_and_info(logger, model, input_dict):
    """
    Thread that updates the logger file with the run status
    by Felipe Ukan - (c) 2019 Lakes Env. Software
    :param logger:
    :param model:
    :param input_dict:
    :return:
    """
    logger.info('Saving metrics and model to logger')
    logger.info(input_dict)
    logger.info(model.to_json())
    index = 0
    metric_to_hist_name = {
        'mae': 'mean_absolute_error',
        'mse': 'mean_squared_error',
        'accuracy': 'accuracy'
    }

    while model:
        index += 1
        time.sleep(60)  # write to file every x seconds
        try:
            logger.info('{0} Current epoch: {1}'.format(index, model.history.epoch[-1]))
            for nn_metric in input_dict['nn_metrics']:
                logger.info('{0} Training {1}: {2:.6f}'.format(index, metric_to_hist_name[nn_metric], round(
                    model.history.history[metric_to_hist_name[nn_metric]][-1], 6)))
                logger.info('{0} Validation {1}: {2:.6f}'.format(index, metric_to_hist_name[nn_metric], round(
                    model.history.history['val_' + metric_to_hist_name[nn_metric]][-1], 6)))
        except Exception as e:
            logger.info('No accuracy or epoch registered at this time. Error: {0}'.format(e))


def rmse(y_true, y_pred):
    """
    Custom RMSE loss function
    by Felipe Ukan - (c) 2019 Lakes Env. Software
    :param y_true:
    :param y_pred:
    :return:
    """
    return utils.keras.backend.sqrt(utils.keras.backend.mean(utils.keras.backend.square(y_pred - y_true), axis=-1))
