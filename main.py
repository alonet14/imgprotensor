
#reference from website: https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
import plaidml.keras
plaidml.keras.install_backend()

from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array

#define model
#demostrate with a very small model with a single LSTM layer contains a single LSTM cell
# one input sample with 3 time step and one feature observed at each time step
t1=0.1
t2=0.2
t3=0.3

inputs1=Input(shape=(3, 1))
lstm1=LSTM(1)(inputs1)
model=Model(inputs=inputs1, output=lstm1)

#define input data
data=array([0.1, 0.2, 0.3]).reshape(1, 3, 1)

#make and show prediction
print(model.predict(data))

