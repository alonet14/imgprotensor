import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop, adam
import matplotlib.pyplot as plt

N = 400
c = 3e8
f = 2e9
lamda = c / f  # wavelength
d = 0.5 * lamda
kappa = np.pi / lamda  # wave number
K = 1  # number of sources
M = 50  # number of ULA elements
snr = 10  # signal to noise ratio
Thetas = np.pi * (np.random.rand(K) - 1 / 2)  # random source directions #pha
array = np.linspace(0, M - 1, M)  # tra ve N mau cach deu nhau trong khoang [0,n/2]
#print(array)
def generatedSignal(theta, M, N, K, f, c):
    S = np.zeros((N, M))
    s = np.zeros((M))
    for n in range(N):
        for m in range(M):
            s[m] = 0
            s[m] += np.exp(1j * m * 2 * np.pi * f / c * np.sin(theta))
            S[n, m] = s[m]
    return S


def array_response_vector(array, theta, d, lamda):  # mang vector phan hoi
    N = array.shape
    v = np.exp(1j * 2 * np.pi * array * np.sin(theta) * d / lamda)
    return v
for i in range(K):
    S = generatedSignal(Thetas[i], M, N, K, f, c)
np.savetxt('a', S)
Y = np.ones((np.size(Thetas, 0), 1))
np.savetxt('b', Y)
print(np.shape(S))
print(Thetas*180 / np.pi)
def load_data():
    data = np.loadtxt("data/data.txt")
    #data = np.loadtxt("a.txt")
    X = data[:, :-1]
    y = data[:, -1:]
    return X, y

# for i in range(N):
#     X = S
#     # print(X)
#     # print("------------")

# if __name__ == "__main__":
#     X, y = load_data()
#     print(np.shape(X))
#     print(np.shape(y))
#     model = Sequential()
#     rbflayer = RBFLayer(10,
#                         initializer=InitCentersRandom(X),
#                         betas=2.0,
#                         input_shape=(1,))
#     model.add(rbflayer)
#     model.add(Dense(1))
#
#     model.compile(loss='mean_squared_error',
#                   optimizer=RMSprop())
#
#     model.fit(X, y,
#               batch_size=50,
#               epochs=2000,
#               verbose=1)
#
#     y_pred = model.predict(X)
#
#     # print(rbflayer.get_weights())
#     # print(model.get_losses_for())
#
#     plt.plot(X, y_pred)
#     plt.plot(X, y)
#     plt.plot([-1, 1], [0, 0], color='black')
#     plt.xlim([-1, 1])
#
#     centers = rbflayer.get_weights()[0]
#     widths = rbflayer.get_weights()[1]
#     plt.scatter(centers, np.zeros(len(centers)), s=20 * widths)
#
#     plt.show()
