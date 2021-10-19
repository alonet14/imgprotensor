from src.controller.data_controller_lstm_v2 import *
from os import environ

environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import tensorflow as tf


# run an experiment
def run_experiment(repeats=10):
    # load data
    prefix = 'D:/project/img_pro_tensorflow/dataset/recorded_data_gen/'
    # prefix = 'D:/project/img_pro_tensorflow/dataset/'

    trainX, trainy, testX, testy = nvh_load_data(prefix)
    # trainX, trainy, testX, testy = load_dataset(prefix)

    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('&gt;#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


# run the experiment
run_experiment()
