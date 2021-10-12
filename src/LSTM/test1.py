import csv

import numpy as np
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
#GRID_NUM_SF = int((doa_max - doa_min) / grid_sf)
GRID_NUM_SF = int((doa_max - doa_min) / grid_sf)
SF_NUM = 6  # number of spatial filters
SF_SCOPE = (doa_max - doa_min) / SF_NUM # spatial scope of each filter
SNR_sf = 10
NUM_REPEAT_SF = 1  # number of repeated sampling with random noise

noise_flag_sf = 1  # 0: noise-free; 1: noise-present
amp_or_phase = 0  # show filter amplitude or phase: 0-amplitude; 1-phase

# # autoencoder parameters
input_size_sf = M * (M-1)
hidden_size_sf = int(1 / 2 * input_size_sf)
output_size_sf = input_size_sf
batch_size_sf = 32
num_epoch_sf = 1000
learning_rate_sf = 0.001

# # training set parameters
#SS_SCOPE = SF_SCOPE / SF_NUM   # scope of signal directions
step_ss = 1  # DOA step (degree) for generating different scenarios
K_ss = 2  # signal number
doa_delta = np.array(np.arange(20) + 1) * 0.1 * SF_SCOPE  # inter-signal direction differences
#doa_delta = np.array(np.arange(121))
SNR_ss = np.array([10, 10, 10]) + 0
NUM_REPEAT_SS = 2  # number of repeated sampling with random noise

noise_flag_ss = 1  # 0: noise-free; 1: noise-present

# # DNN parameters
grid_ss = 1  # inter-grid angle in spatial spectrum
#NUM_GRID_SS = int((doa_max - doa_min + 0.5 * grid_ss) / grid_ss)  # spectrum grids
NUM_GRID_SS = int((doa_max - doa_min + 0.5*grid_ss) / grid_ss)
L = 2  # number of hidden layer
input_size_ss = M * (M - 1)
hidden_size_ss = [int(2 / 3 * input_size_ss), int(4 / 9 * input_size_ss), int(1 / 3 * input_size_ss)]
output_size_ss = int(NUM_GRID_SS / SF_NUM)
batch_size_ss = 32
learning_rate_ss = 0.001
num_epoch_ss = 2

# # test data parameters
test_DOA = np.array([31.5, 41.5])
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

Rho = np.arange(1) * 0.1
num_epoch_test = 1000
RMSE = []

def generate_training_data_ss_AI(M, N, K, d, wavelength, SNR, doa_min, doa_max, step, doa_delta, NUM_REPEAT_SS):
    data_train_ss = {}
    data_train_ss['input_nf'] = []
    data_train_ss['input'] = []
    data_train_ss['target_spec'] = []
    for delta_idx in range(len(doa_delta)):
        delta_curr = doa_delta[delta_idx]  # inter-signal direction differences
        delta_cum_seq_ = [delta_curr]  # doa differences w.r.t first signal
        delta_cum_seq = np.concatenate([[0], delta_cum_seq_])  # the first signal included
        delta_sum = np.sum(delta_curr)  # direction difference between first and last signals
        NUM_STEP = int((doa_max - doa_min - delta_sum) / step)  # number of scanning steps

        for step_idx in range(NUM_STEP):
            doa_first = doa_min + step * step_idx
            DOA = delta_cum_seq + doa_first

            for rep_idx in range(NUM_REPEAT_SS):
                add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
                array_signal = 0
                for ki in range(K):
                    signal_i = 10 ** (SNR[ki] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
                    # phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
                    array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d
                    phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA[ki] / 180 * np.pi)
                    a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
                    # a_i = np.matmul(AP_mtx, a_i)
                    # a_i = np.matmul(MC_mtx, a_i)
                    array_signal_i = np.matmul(a_i, signal_i)
                    array_signal += array_signal_i

                array_output_nf = array_signal + 0 * add_noise  # noise-free output
                array_output = array_signal + 1 * add_noise

                array_covariance_nf = 1 / N * (np.matmul(array_output_nf, np.matrix.getH(array_output_nf)))
                array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
                cov_vector_nf_ = []
                cov_vector_ = []
                for row_idx in range(M):
                    cov_vector_nf_.extend(array_covariance_nf[row_idx, (row_idx + 1):])
                    cov_vector_.extend(array_covariance[row_idx, (row_idx + 1):])
                cov_vector_nf_ = np.asarray(cov_vector_nf_)
                cov_vector_nf_ext = np.concatenate([cov_vector_nf_.real, cov_vector_nf_.imag])
                cov_vector_nf = 1 / np.linalg.norm(cov_vector_nf_ext) * cov_vector_nf_ext
                data_train_ss['input_nf'].append(cov_vector_nf)
                cov_vector_ = np.asarray(cov_vector_)
                cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
                cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
                data_train_ss['input'].append(cov_vector)
                # construct spatial spectrum target
                # target_spectrum = generate_target_spectrum(DOA, doa_min, grid_ss, NUM_GRID_SS)
                #data_train_ss['target_spec'].append(target_spectrum)
                data = [data_train_ss['input']]
                with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    # write multiple rows
                    writer.writerows(data_train_ss['input'])
    print(np.shape(data_train_ss['input']))
    return data_train_ss

a = generate_training_data_ss_AI(M, N, K_ss, d, wavelength, SNR_ss, doa_min, doa_max, step_ss, doa_delta, NUM_REPEAT_SS)