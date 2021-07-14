# from keras.models import load_model
import os


# def _load_nn(network_to_load):
#     """ Loads Neural Network
#     by Felipe Ukan - (c) 2019 Lakes Env. Software
#     :param network_to_load:
#     :return:
#     """
#     # load model
#     try:
#         loaded_model = load_model(network_to_load)
#         print("Loaded model " + network_to_load + " from disk")
#         return loaded_model
#     except Exception as e:
#         print(e)
#         print('Failed to load model')
#         print('Model not found:', network_to_load)
#         exit(2)


def lstm_multout_predict(axisX, network_to_load):
    """ Loads a nn and makes an individual prediction
    by Felipe Ukan - (c) 2019 Lakes Env. Software
    :param axisX:
    :param network_to_load:
    :return:
    """
    # must load network from file
    # must make prediction

    axisX = axisX.reshape(1, -1, axisX.shape[1])
    # print(axisX)

    loaded_model = _load_nn(network_to_load)
    # evaluate loaded model on test data
    # loaded_model.compile(loss='mse', optimizer='nadam', metrics=['accuracy'])
    predict = loaded_model.predict(axisX)
    return predict


def save_nn(model, path_unique_identifier, unique_identifier):
    """ Saves Neural Network
    by Felipe Ukan - (c) 2019 Lakes Env. Software
    :param model:
    :param path_unique_identifier:
    :param unique_identifier:
    :return:
    """
    # save model
    model.save(os.path.join(path_unique_identifier, str(unique_identifier) + '.h5'))
    print("Saved model to " + str(path_unique_identifier))
