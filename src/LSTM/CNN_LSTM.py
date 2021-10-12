import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import keras
import keras.backend as K
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten
from keras.models import Sequential
import tensorflow as tf
import gc
from numba import jit
from IPython.display import display, clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
import sys
sns.set_style("whitegrid")

# time
train_set = pq.read_pandas('train.parquet').to_pandas()

