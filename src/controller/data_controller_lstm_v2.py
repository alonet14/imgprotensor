# lstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
import numpy as np
import os
import src.utils.read_csv_file as rf


# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


def define_label(number):
    index = number - 50
    raw_arr = np.zeros((90))
    raw_arr[index] = 1
    return raw_arr


def read_all_data_and_labeling(prefix=''):
    values = list()
    labels = list()
    for root, _dir, filenames in os.walk(prefix):
        for filename in filenames:
            # print(filename)
            splited_filename = filename.split("_")
            label_from_filename = splited_filename[1].split(".")
            label = label_from_filename[0]
            one_hot_label = define_label(int(label))
            file_path = root + '/' + filename
            data = rf.read_data_trainning_file(file_path)
            value = data.to_numpy()
            rows = value.shape[0]
            for j in range(rows):
                values.append(value[j, 1:])
            for i in range(rows):
                labels.append(one_hot_label)

            # list_data_frame.append(data)
    values = np.asarray(values)
    values=values.reshape((values.shape[0], values.shape[1], 1))
    labels = np.asarray(labels)
    # labels=labels.reshape((labels.shape[0], labels.shape[1], 1))
    return values, labels


def nvh_load_data(prefix=''):
    train_path_prefix = prefix + 'train/'
    test_path_prefix = prefix + 'test/'
    trainX, trainy = read_all_data_and_labeling(train_path_prefix)
    testX, testy = read_all_data_and_labeling(test_path_prefix)
    return trainX, trainy, testX, testy


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1

    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 15, 64
    # n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    n_timesteps=2000
    n_features=1
    n_outputs =90
    # n_features = 1

    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    print('training')
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return 0


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
