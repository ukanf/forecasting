import json
import os
import time
import math
import logging
import datetime
from functools import reduce

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 500)


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
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)


def retrieve_file_with_info(path_for_file_with_info):
    try:
        with open(path_for_file_with_info, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print('Error while loading code to real names dicts: {}'.format(e))
        return {}


def retrieve_unique_unit_code(path_for_file_with_metadata, var_name):
    # todo logger so we know if it has too many
    parameter_code, poc, dt_type = var_name.split('-')
    try:
        with open(path_for_file_with_metadata, 'r') as file:
            data = json.load(file)
            unique_unit_code = list(data[parameter_code + '-' + poc]['unit'].keys())
            if len(unique_unit_code) != 1:
                return ''
            else:
                return unique_unit_code[0]
    except Exception as e:
        print('')
        return ''


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


def resolve_keras_callbacks(v, unique_identifier, path_unique_identifier):
    """
    :param v:
    :param unique_identifier:
    :param path_unique_identifier:
    :return:
    """
    logger = logging.getLogger(unique_identifier)
    keras_callbacks = None

    if v.callbacks and len(v.callbacks) > 0:
        keras_callbacks = []
        if 'tensorboard' in v.callbacks:
            logger.info('Adding Tensorboard to callbacks')
            keras_callbacks.append(
                keras.callbacks.TensorBoard(log_dir=os.path.join(path_unique_identifier, 'tensor_board'),
                                            histogram_freq=v.batchsize,
                                            write_graph=True,
                                            write_images=True,
                                            write_grads=True,
                                            update_freq=v.epochs)
            )
        if 'earlystopping' in v.callbacks:
            logger.info('Adding Early Stopping to callbacks')
            keras_callbacks.append(EarlyStopping(monitor='val_loss', restore_best_weights=True,
                                                 patience=250, mode='min', verbose=1))

    # for now we reset.. the arguments given do not meet any of the expected
    if keras_callbacks is not None and len(keras_callbacks) == 0:
        logger.warning('Callbacks declared, but no callback added to the model!')
        keras_callbacks = None
    return keras_callbacks


# convert an array of values into a dataset matrix
def _create_3d_lookback_array(data, look_back):
    """
    Creates a 3 dimensional array for our LSTM/GRU networks.
    Dimensions are: (num samples, num time steps, num features). For example, we can have a output with 1024 rows, 25 lookback units (current + part 24 lookbacks), and O3 shifted 24 hours as input, i.e. (1024, 25, 1)
    this method creates a input shape of: (num samples, timesteps, num parameters to predict)
    by Felipe Ukan - 
    :param data: actual array with all the data
    :param look_back: int number that tells how many timesteps behind to look
    """

    # determine number of data samples
    rows_data, cols_data = np.shape(data)

    # determine number of row to iterate
    tot_batches = int(rows_data)

    # initializes 3D tensor
    threed_tensor = np.zeros((tot_batches - look_back, look_back + 1, cols_data))

    # populate 3D tensor
    for sample_num in range(look_back, tot_batches):
        try:
            threed_tensor[sample_num - look_back, :, :] = data[(sample_num - look_back):sample_num + 1, :]
        except Exception as e:
            print('ERROR: {}. Not able to add current element to the threeD array'.format(e))

    return threed_tensor


def remove_other_parameters_cols(df, parameters):
    """
    Removes columns from df that are not in parameters
    :param df:
    :param parameters:
    :return:
    """
    for col in df.columns:
        # print col.split('_')[1]
        if col not in parameters:
            del df[col]


def create_XY_arrays_multout(df, unique_identifier, v):
    """
    Creates input and output arrays.
    by Felipe Ukan - 
    :param unique_identifier:
    :param df: dataframe with the data
    :param v: namespace with all the variable values for NN (here we unpack the ones we will use)
    :return: tuple, where the first (dataset1) is the input for the NN and the second (dataplot1) is the expected output time_steps ahead from the input.
    """
    logger = logging.getLogger(unique_identifier)

    # unpacks parameters that will be used in this method
    look_back = v.lookback
    output_vars = v.output_vars
    input_vars = v.input_vars
    time_steps = v.timesteps
    use_days_of_week = v.use_days_of_week
    use_hours_of_day = v.use_hours_of_day
    num_buckets = v.num_buckets
    min_samples_to_train = v.min_samples_to_train

    # initializes variables
    skip_ahead = max(time_steps) + look_back

    # we have the parameters we are not going to use and then drop the rows with NaN
    # send the df and a set of all input and output var names
    remove_other_parameters_cols(df, {*input_vars, *output_vars})
    df = df.dropna()

    if num_buckets:
        for output_var in output_vars:
            df[output_var] = pd.cut(df[output_var], bins=num_buckets, labels=np.arange(num_buckets), right=False)
        for input_var in input_vars:
            df[input_var] = pd.cut(df[input_var], bins=num_buckets, labels=np.arange(num_buckets), right=False)

    target_vars_df = pd.DataFrame(index=df.index)
    input_var_df = pd.DataFrame(index=df.index)

    if use_hours_of_day:
        input_var_df['hour_of_day'] = [int(row.hour) for row in pd.to_datetime(df.index)]
        input_var_df = input_var_df.join(pd.get_dummies(input_var_df['hour_of_day'], prefix='h'))  # one hot encoding
        del input_var_df['hour_of_day']

    if use_days_of_week:
        # we have to do this because of daylight savings and other things that can break getting weekday directly from day
        input_var_df['day_of_week'] = [datetime.datetime.strptime(row[0:10], '%Y-%M-%d').weekday() for row in df.index]
        input_var_df = input_var_df.join(pd.get_dummies(input_var_df['day_of_week'], prefix='d'))  # one hot encoding
        del input_var_df['day_of_week']

    for output_var in output_vars:
        try:
            for time_step in time_steps:
                target_vars_df[output_var + '_t+' + str(time_step)] = df[output_var].shift((time_step + look_back) * -1)
        except KeyError as e:
            print('Not a valid key for output parameter: ', e)
            print('Valid inputs are: ', output_vars)
            exit()

    for input_var in input_vars:
        try:
            if input_var not in input_var_df.columns:
                input_var_df[input_var] = df[input_var]
        except KeyError as e:
            print('Not a valid key for extra input parameter: ', e)
            print('Valid inputs are: ', df.columns)
            exit()

    # here axis variables become numpy arrays
    axisX = input_var_df.values
    axisX = axisX[:-skip_ahead]  # need to trim last values that are now 0 for predicted..
    axisY = target_vars_df.values
    axisY = axisY[:-skip_ahead]

    if len(axisX) < min_samples_to_train or len(axisY) < min_samples_to_train:
        logger.critical('NOT ENOUGH DATA TO CONTINUE either {} or {} are < {}'.format(len(axisX), len(axisY), min_samples_to_train))
        exit()

    return axisX, axisY, input_var_df.columns, target_vars_df.columns


def resolve_scaler(axisX, axisY, input_vars_names, output_vars_names, unique_identifier, path_unique_identifier, v):
    """
    :param axisX:
    :param axisY:
    :param input_vars_names:
    :param output_vars_names:
    :param unique_identifier:
    :param path_unique_identifier:
    :param v:
    :return:
    """
    logger = logging.getLogger(unique_identifier)

    scalerY = False

    if v.scaler == 'MinMaxScaler':
        print(v)
        print(v.in_absolute_filename_minmax)
        if not v.in_absolute_filename_minmax:
            scalerX = MinMaxScaler(feature_range=(0, 1))
            axisX = scalerX.fit_transform(axisX)
            scalerY = MinMaxScaler(feature_range=(0, 1))
            axisY = scalerY.fit_transform(axisY)
            # saving scaler
            logger.info('Saving scaler')
            scaler_filename = os.path.join(path_unique_identifier, 'MinMaxScaler.save')
            joblib.dump(scalerY, scaler_filename)
        else:
            # here we know that we already applied min max scaler before
            # so we load it and fit the scaler accordingly
            df = pd.read_csv(v.in_absolute_filename_minmax)
            scalerY = MinMaxScaler(feature_range=(0, 1))
            scalersY = {
                "O3": [[df["O3"][0]], [df["O3"][1]]],
                "SO2": [[df["SO2"][0]], [df["SO2"][1]]],
                "NO2": [[df["NO2"][0]], [df["NO2"][1]]],
                "CO": [[df["CO"][0]], [df["CO"][1]]],
                "PM10": [[df["PM10"][0]], [df["PM10"][1]]],
            }
            scalerY.fit(scalersY.get("O3"))
    elif v.scaler == 'RobustScaler':
    # todo ROBUST SCALER MUST BE APPLIED ONLY TO CONTINUOUS COLUMNS
        scalerX = RobustScaler()
        axisX = scalerX.fit_transform(axisX)
        scalerY = RobustScaler()
        axisY = scalerY.fit_transform(axisY)
        logger.info('Saving scaler')
        scaler_filename = os.path.join(path_unique_identifier, 'RobustScaler.save')
        joblib.dump(scalerY, scaler_filename)
    else: logger.warning('Not using a scaler.')

    return axisX, axisY, scalerY


# def get_stateful_dataset(dataset, batchsize):
#     """
#     Trims dataset to make it stateful (need to be divisible by the batchsize)
#     by Felipe Ukan - 
#     :param dataset:
#     :param batchsize:
#     :return:
#     """
#     while len(dataset) % batchsize != 0:
#         dataset = dataset[:len(dataset)-1]
#
#     return dataset

def prepare_XY_arrays(axisX, axisY, unique_identifier, v):
    """
    Creates training and test sets
    by Felipe Ukan - 
    :param unique_identifier:
    :param axisX:
    :param axisY:
    :param v:
    :return:
    """
    logger = logging.getLogger(unique_identifier)

    # unpacks variables we use here
    train_split = v.train_split
    look_back = v.lookback
    batchsize = v.batchsize
    train_size = int(len(axisX) * train_split)
    # test_size = len(axisX) - train_size
    train, test = axisX[0:train_size], axisX[train_size:len(axisX)]

    # prepare output arrays
    trainY, testY = axisY[1:train_size], axisY[train_size+1:len(axisY)]

    # todo is this still necessary??
    n, p = np.shape(trainY)
    if n < p:
        trainY = trainY.T
        testY = testY.T

    # resize input sets
    trainX1 = train[:len(trainY), ]
    testX1 = test[:len(testY), ]

    # prepare input Tensors
    trainX = _create_3d_lookback_array(trainX1, look_back)
    testX = _create_3d_lookback_array(testX1, look_back)

    # trims target arrays to match input lengths
    if len(trainX) < len(trainY):
        trainY = np.asmatrix(trainY[:len(trainX)])

    if len(testX) < len(testY):
        testY = np.asmatrix(testY[:len(testX)])

    # following lines ensure that we do not have samples with nan
    trainX, trainY = clean_nan_from_samples(trainX, trainY, v)
    testX, testY = clean_nan_from_samples(testX, testY, v)

    return trainX, testX, trainY, testY


def clean_nan_from_samples(base_dataset, corresponding_dataset, v):
    """
    Ensures that we do not have samples with nan.
    :param base_dataset: Array with
    :param corresponding_dataset:
    :param v:
    :return:
    """
    base_copy = base_dataset.copy()
    aux = 0
    for index, sample in enumerate(base_copy):
        if np.isnan(sample.flatten()).any():
            # print('{} has nan'.format(index))
            base_dataset = np.delete(base_dataset, index - aux, 0)
            corresponding_dataset = np.delete(corresponding_dataset, index - aux, 0)
            aux += 1

    mirror_copy = corresponding_dataset.copy()
    aux = 0
    for index, sample in enumerate(mirror_copy):
        if np.isnan(sample.flatten()).any():
            # print('{} has nan'.format(index))
            base_dataset = np.delete(base_dataset, index - aux, 0)
            corresponding_dataset = np.delete(corresponding_dataset, index - aux, 0)
            aux += 1

    return np.array(base_dataset), np.array(corresponding_dataset)


def read_csvdata(v, unique_identifier, skipfooter=0):
    """
    Creates dataframe from csv file
    by Felipe Ukan - 
    :param v:
    :param unique_identifier:
    :param skipfooter:
    :return:
    """
    logger = logging.getLogger(unique_identifier)
    df = None

    try:
        df = pd.read_csv(v.in_absolute_filename_dataset, engine='python', skipfooter=skipfooter)
        df = df.set_index(df.columns[0])
        df.index.rename('id', inplace=True)
    except Exception as e:
        print('Error: {}; while trying to read file'.format(e))
        exit()

    print('dataset used:', v.in_absolute_filename_dataset)

    # verifications that predict_var exists
    for output_var in v.output_vars:
        if output_var not in df.columns:
            logger.error('Cannot continue.. predict var not in columns')
            logger.critical('Predict var not in columns', output_var)
            logger.critical('The Valid parameters are', df.columns)
            exit()

    return df


def create_modelloss_graph(history, savefile_path=False):
    """Generates and saves model loss graph
    by Felipe Ukan - 
    :param history:
    :param savefile_path:
    :return:
    """
    plt.title('Model loss')
    plt.plot(history.history['val_loss'], label='validation')
    plt.plot(history.history['loss'], label='training')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    if savefile_path:
        plt.savefig(str(savefile_path) + '.png')
    plt.close()


def create_realpredict_with_fill_graph(testY, testPredict, mae_result, var_name='Title', savefile_path=False, x_axis_label=False, y_axis_label=False, x_label=False, y_label=False, vertical_hourly_num_steps=None, off_set_vertical_hourly_num_steps=0):
    """
    TODO: change this method to receive only a dictionary and unpack the necessary vars inside
    Creates real data and predicted data graph and fills around the real data the MAE error (expected mean absolute error)
    by Felipe Ukan - 
    :param testY:
    :param testPredict:
    :param mae_result:
    :param var_name:
    :param savefile_path:
    :param x_axis_label:
    :param y_axis_label:
    :param x_label:
    :param y_label:
    :param vertical_hourly_num_steps:
    :param off_set_vertical_hourly_num_steps:
    :return:
    """
    x_axis_label = str(x_axis_label) if x_axis_label else ''
    y_axis_label = str(y_axis_label) if y_axis_label else ''
    x_label = str(x_label) if x_label else 'real data'
    y_label = str(y_label) if y_label else 'prediction'
    # plot baseline and predictions
    plt.close('all')
    # plt.gca().set_ylim([0, 0.1])
    plt.plot(testY, label=x_label)
    plt.plot(testPredict, label=y_label)
    # rolling_std = pd.Series(testPredict).rolling(5).std()
    # pd.rolling_std(testPredict, 20)
    # print(pd.Series(testPredict))
    # print(rolling_std)
    plt.fill_between(range(len(testPredict)), testY-mae_result, testY+mae_result, color='b', alpha=0.2)
    plt.title(str(var_name))
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.grid()

    if vertical_hourly_num_steps and 0 <= off_set_vertical_hourly_num_steps < 12:
        plt.axvline(x=off_set_vertical_hourly_num_steps, color='green', linestyle=':', label='Midnight')
        plt.axvline(x=12+off_set_vertical_hourly_num_steps, color='red', linestyle=':', label='Noon')
        for xc in range(24, vertical_hourly_num_steps, 24):
            plt.axvline(x=xc, color='green', linestyle=':')
            plt.axvline(x=xc+12, color='red', linestyle=':')

    plt.legend(loc='lower right')
    if savefile_path:
        plt.savefig(str(savefile_path) + '.png')
    plt.show()
    plt.close()


def create_realpredict_graph(testY, testPredict, title='', savefile_path=False, x_axis_label=False, y_axis_label=False, label_one=False, label_two=False):
    """
    TODO: change arguments to receive a dictionary
    Creates real data and predicted data graph
    by Felipe Ukan - 
    :param testY:
    :param testPredict:
    :param title:
    :param savefile_path:
    :param x_axis_label:
    :param y_axis_label:
    :param label_one:
    :param label_two:
    :return:
    """
    x_axis_label = str(x_axis_label) if x_axis_label else ''
    y_axis_label = str(y_axis_label) if y_axis_label else ''
    label_one = str(label_one) if label_one else 'Measured data'
    label_two = str(label_two) if label_two else 'Predicted data'
    # plot baseline and predictions
    plt.close('all')
    # plt.gca().set_ylim([-5, 60])
    plt.plot(testY, label=label_one)
    plt.plot(testPredict, label=label_two)
    plt.title(str(title))
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.grid(alpha=0.25)
    plt.legend(loc='lower right')
    if savefile_path:
        plt.savefig(str(savefile_path) + '.png')
    else:
        plt.show()
    plt.close()


def save_realpredict_data(testY, testPredict, output_save_path):
    """
    Saves the real and predicted data as a csv file
    by Felipe Ukan - 
    :param testY:
    :param testPredict:
    :param output_save_path:
    :return:
    """
    new_testY = testY.reshape(-1)
    for x in range(len(testPredict) - len(new_testY)):
        new_testY = np.append(new_testY, np.nan)
    new_testPredict = testPredict.reshape(-1)

    data = {
        'testY': new_testY,
        'testPredict': new_testPredict
    }
    df = pd.DataFrame(index=range(len(new_testPredict)), data=data)
    df.to_csv(output_save_path, float_format='%g', na_rep='NA')

    return output_save_path


def calculate_MAE(real_values, predicted_values, label=None):
    """
    Calculates Mean Absolute Error <https://en.wikipedia.org/wiki/Mean_absolute_error>
    by Felipe Ukan - 
    :param real_values:
    :param predicted_values:
    :param label:
    :return:
    """
    return mean_absolute_error(real_values, predicted_values)


def calculate_NMAE(real_values, predicted_values, min_value, max_value, label=None):
    """
    Calculates Normalized Mean Absolute Error <https://en.wikipedia.org/wiki/Mean_absolute_error>
    by Felipe Ukan - 
    :param real_values:
    :param predicted_values:
    :param label:
    :return:
    """
    return calculate_MAE(real_values, predicted_values, label) / (max_value - min_value)


def calculate_RMSE(real_values, predicted_values, label=None):
    """
    Calculates Root Mean Squared Error <https://en.wikipedia.org/wiki/Root-mean-square_deviation>
    by Felipe Ukan - 
    :param real_values:
    :param predicted_values:
    :param label:
    :return:
    """
    return math.sqrt(mean_squared_error(real_values, predicted_values))


def calculate_NRMSE(real_values, predicted_values, min_value, max_value, label=None):
    """
    Calculates Normalized Root Mean Squared Error <https://en.wikipedia.org/wiki/Root-mean-square_deviation>
    by Felipe Ukan - 
    :param real_values:
    :param predicted_values:
    :param label:
    :return:
    """
    return calculate_RMSE(real_values, predicted_values, label) / (max_value - min_value)


# def print_metrics(metrics):
#     """
#     :param metrics: dict
#     :return:
#     """
#
#     for key, value in metrics.items():
#         print(key, ': ', value)
#
#
# def save_print_to_file(unique_identifier):
#     import sys
#     f = open('results/' + unique_identifier + '.txt', 'w')
#     sys.stdout = f

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
