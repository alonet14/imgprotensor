import numpy as np
from GenerateData import *
import pandas as pa
import scipy.linalg as la
from math import sqrt
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

N_sample = np.array(range(36))
M = 10
K = 1
f = 2e9
c = 3e8
data = []
for i in range(-60, 60, 1):
    a = 0
    a = a + 1
    Theta_train = np.array(np.arange(i, i + 1, 0.02))
    Theta_train = Theta_train.reshape((np.size(Theta_train),1))

    # LabelTr = np.ones((np.size(Theta_train, 0), 1))*a
    LabelTr =Theta_train
    N = np.size(Theta_train, 0)

    S_train = generatedSignal(Theta_train, M, N, K, f, c)
    np.savetxt('dt', S_train)
    ThetaTs = np.array(np.arange(10, 30.09, 0.01))
    ThetaTs = ThetaTs.reshape((np.size(ThetaTs),1))

    S_Test = generatedSignal(ThetaTs, M, np.size(ThetaTs, 0), K, f, c)
    LabelTs = np.ones((np.size(ThetaTs, 0), 1))*a
    model = Sequential()
    S_train = S_train.reshape(S_train.shape[0], 1, S_train.shape[1])
    data.append(S_train)
    print(S_train)
    model.add(LSTM(4, batch_input_shape=(1, S_train.shape[1], S_train.shape[2]), stateful=True))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(10):
        model.fit(S_train, LabelTr, epochs=10, batch_size=1, verbose=True, shuffle=False)
        model.reset_states()
    model.save_weights('a.h5')

model.load_weights('a.h5')
S_train1 = generatedSignal(20, M, N, K, f, c)
print(model.predict(S_train1,batch_size=1)*20)
print(LabelTr)
# model = Sequential()
# S_train = S_train.reshape(S_train.shape[0], 1, S_train.shape[1])
# model.add(LSTM(4 , batch_input_shape=(2, S_train.shape[1], S_train.shape[2]), stateful=True))
# model.add(Dense(1, activation='softmax'))
# model.compile(loss='mean_squared_error', optimizer='adam')
# for i in range(10):
#     model.fit(S_train, LabelTr, epochs=10, batch_size=2, verbose=True, shuffle=False)
#     model.reset_states()