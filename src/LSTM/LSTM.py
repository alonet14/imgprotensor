# import pandas as pa
import scipy.linalg as la
from math import sqrt

from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
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
GRID_NUM_SF = int((doa_max - doa_min) / grid_sf)
SF_NUM = 6  # number of spatial filters
SF_SCOPE = (doa_max - doa_min) / SF_NUM  # spatial scope of each filter
SNR_sf = 10
NUM_REPEAT_SF = 1  # number of repeated sampling with random noise

noise_flag_sf = 1  # 0: noise-free; 1: noise-present
amp_or_phase = 0  # show filter amplitude or phase: 0-amplitude; 1-phase

# # autoencoder parameters
input_size_sf = M * (M - 1)
hidden_size_sf = int(1 / 2 * input_size_sf)
output_size_sf = input_size_sf
batch_size_sf = 32
num_epoch_sf = 1000
learning_rate_sf = 0.001
step_ss = 1  # DOA step (degree) for generating different scenarios
K_ss = 2  # signal number
doa_delta = np.array(np.arange(20) + 1) * 0.1 * SF_SCOPE  # inter-signal direction differences
SNR_ss = np.array([10, 10, 10]) + 0
NUM_REPEAT_SS = 10  # number of repeated sampling with random noise
# # training set parameters
# SS_SCOPE = SF_SCOPE / SF_NUM   # scope of signal directions
step = 1  # DOA step (degree) for generating different scenarios
K = 2  # signal number
doa_delta = np.array(np.arange(26) + 1) * 0.1 * SF_SCOPE  # inter-signal direction differences
SNR = np.array([10, 10, 10]) + 0
SNR1 = 10
NUM_REPEAT_SS = 10  # number of repeated sampling with random noise

noise_flag_ss = 1  # 0: noise-free; 1: noise-present

# # DNN parameters
grid_ss = 1  # inter-grid angle in spatial spectrum
NUM_GRID_SS = int((doa_max - doa_min + 0.5 * grid_ss) / grid_ss)  # spectrum grids

# # test data parameters
test_DOA = np.array([30.5, 40.5])
test_K = len(test_DOA)
test_SNR = np.array([10, 10])

# # retrain the networks or not
reconstruct_nn_flag = True
retrain_sf_flag = True
retrain_ss_flag = True

# # file path of neural network parameters
model_path_nn = 'initial_model_AI.npy'
model_path_sf = 'spatialfilter_model_AI.npy'
model_path_ss = 'spatialspectrum_model_AI.npy'

# # array imperfection parameters
mc_flag = True
ap_flag = False
pos_flag = False


rmse_path = 'arrayimperf'
if mc_flag == True:
    rmse_path += '_mc'
if ap_flag == True:
    rmse_path += '_ap'
if pos_flag == True:
    rmse_path += '_pos'
rmse_path += '.npy'

rho = np.arange(1) * 0.1
if mc_flag == True:
    mc_para = rho * 0.3 * np.exp(1j * 60 / 180 * np.pi)
    MC_coef = mc_para ** np.array(np.arange(M))
    MC_mtx = la.toeplitz(MC_coef)
else:
    MC_mtx = np.identity(M)
# amplitude & phase error
if ap_flag == True:
    amp_coef = rho * np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.2, -0.2, -0.2, -0.2, -0.2])
    phase_coef = rho * np.array([0.0, -30, -30, -30, -30, -30, 30, 30, 30, 30])
    AP_coef = [(1 + amp_coef[idx]) * np.exp(1j * phase_coef[idx] / 180 * np.pi) for idx in range(M)]
    AP_mtx = np.diag(AP_coef)
else:
    AP_mtx = np.identity(M)
# sensor position error
if pos_flag == True:
    pos_para_ = rho * np.array([0.0, -1, -1, -1, -1, -1, 1, 1, 1, 1]) * 0.2 * d
    pos_para = np.expand_dims(pos_para_, axis=-1)
else:
    pos_para = np.zeros([M, 1])

output_size = 90
grid = 1
GRID_NUM = 120
# data_train_sf = generate_training_data_ss_AI(M, N, K_ss, d, wavelength, SNR_ss, doa_min, doa_max, step_ss, doa_delta,
#                                                 NUM_REPEAT_SS, grid_ss, NUM_GRID_SS, MC_mtx, AP_mtx, pos_para)

X_train = np.loadtxt('data')
y = np.loadtxt('label')
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
print(X_train.shape[0], X_train.shape[1])
print(np.shape(X_train))

# #Model
# model = Sequential()
# # model.add(LSTM(90, input_shape = (X_train.shape[0], X_train.shape[1])))
# model.add(LSTM(256, input_shape= (X_train.shape[1], X_train.shape[2])))
# model.add(Dropout(0.2))
# # model.add(Dense(y.shape[1], activation='softmax'))
# model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()
# filepath = 'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callback_list = [checkpoint]
# model.fit(X_train, y, epochs=100, batch_size=64, validation_split=0.33, callbacks=callback_list, verbose=1)
