import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.linalg as la
from GenerateData_SG import *
# import tensorflow.compat.v1 as tf
import tensorflow as tf1
from keras.optimizers import adam
from tensorflow.python.keras.optimizers import Adam
from Model_LSTM import *

# tf.disable_v2_behavior()

# Para
# # array signal parameters
fc = 2e9  # Tần số tín hiệu đến
c = 3e8  # Vận tốc ánh sáng
M = 10  # Số phần tử anten
N = 400  # snapshot number
wavelength = c / fc  # Bước sóng
d = 0.5 * wavelength  # Khoảng cách giữa các phần tử anten
K_ss = 2  # signal number
doa_min = -60  # DOA max (degree)
doa_max = 60  # DOA min (degree)
# # spatial filter training parameters

grid_sf = 1  # DOA step (degree) for generating different scenarios
# GRID_NUM_SF = int((doa_max - doa_min) / grid_sf)
GRID_NUM_SF = int((doa_max - doa_min) / grid_sf)
SF_NUM = 6  # number of spatial filters
SF_SCOPE = (doa_max - doa_min) / SF_NUM  # spatial scope of each filter
SNR_sf = 10
NUM_REPEAT_SF = 1  # number of repeated sampling with random noise

noise_flag_sf = 1  # 0: noise-free; 1: noise-present
amp_or_phase = 0  # show filter amplitude or phase: 0-amplitude; 1-phase
NUM_GRID_SS = 121
# # training set parameters
# SS_SCOPE = SF_SCOPE / SF_NUM   # scope of signal directions
step_ss = 1  # DOA step (degree) for generating different scenarios

doa_delta = np.array(np.arange(20) + 1) * 0.1 * SF_SCOPE  # inter-signal direction differences
SNR_ss = np.array([10, 10, 10]) + 0
NUM_REPEAT_SS = 10  # number of repeated sampling with random noise

noise_flag_ss = 1  # 0: noise-free; 1: noise-present

# # DNN parameters
grid_ss = 1  # inter-grid angle in spatial spectrum

input_size_ss = M * (M - 1)  # 90
batch_size_ss = 32
learning_rate_ss = 0.001
num_epoch_ss = 1
n_hidden = 256
n_classes = 121

weights = {
    'hidden': tf.Variable(tf.random_normal([input_size_ss, n_hidden])),  # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Train
data_train_ss = generate_training_data_ss_AI(M, N, K_ss, d, wavelength, SNR_ss, doa_min, doa_max, step_ss, doa_delta,
                                             NUM_REPEAT_SS, grid_ss, NUM_GRID_SS)
enmod_2 = Ensemble_Model(input_size_ss, weights, biases, n_hidden)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print('Training...')
    # train
    for epoch in range(num_epoch_ss):
        [data_batches, label_batches] = generate_spec_batches(data_train_ss, batch_size_ss, noise_flag_ss)
        print(np.shape(data_batches))
        for batch_idx in range(len(data_batches)):
            data_batch = data_batches[batch_idx]
            label_batch = label_batches[batch_idx]
            feed_dict = {enmod_2.data_train_: data_batch, enmod_2.label_ss_: label_batch}
            _, loss_ss = sess.run([enmod_2.train_op_ss, enmod_2.loss_ss], feed_dict=feed_dict)

            print('Epoch: {}, Batch: {}, loss: {:g}'.format(epoch, batch_idx, loss_ss))

# Test
test_DOA = np.array([55, 50])
test_K = len(test_DOA)
test_SNR = np.array([10, 10])
num_epoch_test = 1000
RMSE = []
tf.reset_default_graph()


# test_cov_vector = generate_array_cov_vector_AI(M, N, d, wavelength, test_DOA, test_SNR)
# test_cov_vector = test_cov_vector.reshape(1, test_cov_vector.shape[0])
# print(np.shape(test_cov_vector))
# print(test_cov_vector)
# print(type(test_cov_vector))
def generate_spec_batches_AI(data_train, batch_size, noise_flag):
    if noise_flag == 0:
        data_ = data_train['input_nf']
    else:
        data_ = data_train
    label_ = data_train
    data_len = 1

    # shuffle data
    shuffle_seq = np.random.permutation(range(data_len))
    data = [data_[idx] for idx in shuffle_seq]
    label = [label_[idx] for idx in shuffle_seq]

    # generate batches
    num_batch = int(data_len / batch_size)
    data_batches = []
    label_batches = []
    for batch_idx in range(num_batch):
        batch_start = batch_idx * batch_size
        batch_end = np.min([(batch_idx + 1) * batch_size, data_len])
        data_batch = data[batch_start: batch_end]
        label_batch = label[batch_start: batch_end]
        data_batches.append(data_batch)
        label_batches.append(label_batch)

    return data_batches, label_batches


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print('testing...')

    # # test
    est_DOA = []
    MSE_rho = np.zeros([test_K, ])
    for epoch in range(num_epoch_test):
        test_cov_vector = generate_array_cov_vector_AI(M, N, d, wavelength, test_DOA, test_SNR)
        data_batch = np.expand_dims(test_cov_vector, axis=-1)
        print(data_batch.shape)

        data_batch = data_batch.tolist()
        print(type(data_batch))
        feed_dict = {enmod_2.data_train: data_batch}
        ss_output = sess.run(enmod_2.output_ss, feed_dict=feed_dict)
        ss_min = np.min(ss_output)
        ss_output_regularized = [ss if ss > -ss_min else [0.0] for ss in ss_output]

        est_DOA_ii = get_DOA_estimate(ss_output, test_DOA, doa_min, grid_ss)
        est_DOA.append(est_DOA_ii)
        MSE_rho += np.square((est_DOA_ii - test_DOA))

    RMSE_rho = np.sqrt(MSE_rho / (num_epoch_test))
    RMSE.append(RMSE_rho)

print("Act_DoA", test_DOA)
print("est_DoA", est_DOA)
# print("----------")
# print('RMSE: ', RMSE)
# print('RMSE mean: ', np.mean(RMSE))
