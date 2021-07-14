import pandas as pd
from xgboost import XGBClassifier
import models.utils as utils
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# *** Implementing ***
# CALLS TRIER WITH PARAMETERS WE WANT (overwrites default params)
trier_params = {
    'in_filename_datasets_path': 'datasets/refined_datasets/1549317056/dataset.csv',  # str
    'prefix_id': 'california-13-02-2019-Shafter',
    'input_var': ['42603_6001'],  # only used as input
    'output_vars': ['42603_6001'],  # only used as output
    'Comments': ''
}


def create_my_temp_plot(df):
    values = df.values

    pyplot.plot(values)

    pyplot.xlabel('temp')
    pyplot.title('My temp')
    pyplot.ylabel('y label')

    pyplot.show()
    pyplot.figure()
    # pyplot.savefig(str(i))


def my_shifter(df):
    for time_step in range(0, 25):
        df['lookback' + str(time_step)] = df['42603_6001'].shift(time_step)

    df = df.dropna()
    return df


if __name__ == "__main__":
    df = utils.read_csvdata(trier_params['in_filename_datasets_path'], skipfooter=0)
    bins = 150
    df = df[trier_params['input_var']]  # we will only work with ozone for now
    # create_my_temp_plot(df)
    df['42603_6001'] = pd.cut(df['42603_6001'], bins=bins, labels=np.arange(bins), right=False)

    df = my_shifter(df)

    # create_my_temp_plot(df)
    # print(df.head(5))
    predict_timesteps = 1

    # split data into X and y
    X = df.values[:len(df.values) - predict_timesteps, 1:len(df.columns)-1]
    Y = df.values[predict_timesteps:, 0]

    # print(X[:5])
    # print(Y[:5])
    # exit()
    test_size = 0.05

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=False)

    model = XGBClassifier(max_depth=5,
                          min_child_weight=1,
                          learning_rate=0.1,
                          n_estimators=50,
                          silent=True,
                          booster='gbtree',
                          objective='binary:logistic',
                          gamma=1,
                          max_delta_step=1,
                          subsample=1,
                          colsample_bytree=1,
                          colsample_bylevel=1,
                          reg_alpha=1,
                          reg_lambda=1,
                          scale_pos_weight=15,
                          seed=1,
                          missing=None)
    model.fit(X_train, y_train)

    print(model)

    y_pred = model.predict(X_test)
    # predictions = [round(value) for value in y_pred]
    predictions = list(y_pred)

    utils.create_realpredict_graph(y_test[:300], predictions[:300])

    print('MAE: ', utils.calculate_MAE(y_test, predictions))
    # print("Accuracy: %.2f%%" % (accuracy_score(list(y_test), predictions) * 100.0))

    exit()
