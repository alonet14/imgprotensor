import pandas as pa
import scipy.linalg as la
from math import sqrt
import tensorflow.compat.v1 as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Bidirectional
from keras.layers import LSTM
import numpy as np
from GenerateData_SG import *

fc = 2e9  # carrier frequency
c = 3e8  # light speed
M = 10  # array sensor number
N = 400  # snapshot number
wavelength = c / fc  # signal wavelength
d = 0.5 * wavelength  # inter-sensor distance

# # spatial filter training parameters
doa_min = -60  # minimal DOA (degree)
doa_max = 60  # maximal DOA (degree)
grid_sf = 1  # DOA step (degree) for generating different scenarios
SF_NUM = 6  # number of spatial filters

step_ss = 1  # DOA step (degree) for generating different scenarios
K_ss = 2  # signal number
#doa_delta = np.array(np.arange(20) + 1) * 0.1  # inter-signal direction differences
SNR_ss = np.array([10, 10, 10]) + 0
NUM_REPEAT_SS = 10  # number of repeated sampling with random noise
# # training set parameters
# SS_SCOPE = SF_SCOPE / SF_NUM   # scope of signal directions
step = 1  # DOA step (degree) for generating different scenarios
K = 2  # signal number
doa_delta = np.array(np.arange(20) + 1) * 0.01*SF_NUM  # inter-signal direction differences
SNR = np.array([10, 10, 10]) + 0
SNR1 = 10
NUM_REPEAT_SS = 10  # number of repeated sampling with random noise

noise_flag_ss = 1  # 0: noise-free; 1: noise-present

# # DNN parameters
grid_ss = 1  # inter-grid angle in spatial spectrum
NUM_GRID_SS = 121  # spectrum grids

# # test data parameters
test_DOA = np.array([30.5, 40.5])
test_K = len(test_DOA)
test_SNR = np.array([10, 10])
data_train_sf = generate_training_data_ss_AI(M, N, K, d, wavelength, SNR, doa_min, doa_max, step, doa_delta, NUM_REPEAT_SS, grid_ss,
                                 NUM_GRID_SS)

X_train = np.loadtxt('data')
y = np.loadtxt('label')
y = np.asarray(y)
print(y.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
print(X_train.shape[0], X_train.shape[1])
print(np.shape(X_train))
print(y.shape[1])
batch_size = 32
# #Model
model = Sequential()
# model.add(LSTM(90, input_shape = (X_train.shape[0], X_train.shape[1])))
model.add(LSTM(256, input_shape= (X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
filepath = 'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
callback_list = [checkpoint]
for i in range(10):
    model.fit(X_train, y, epochs=10, batch_size=64, validation_split=0.33, callbacks=callback_list, verbose=1)
    model.reset_states()

