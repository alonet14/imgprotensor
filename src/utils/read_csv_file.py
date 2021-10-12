import pandas as pd
import numpy as np

def read_data_trainning_file(file_path):
    with open(file_path, 'r') as f:
        rs = pd.read_csv(f)
        f.close()
    return rs


