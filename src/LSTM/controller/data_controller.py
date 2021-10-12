import os
from pathlib import Path
import numpy as np
import src.utils.read_csv_file as rf
import pandas as pd
import json

# DUONG DAN DEN THU MUC CHUA DU LIEU
PROJECT_PATH = Path(__file__).parent.parent.parent.parent
CONFIG_DATA_FOLDER = str(PROJECT_PATH) + '\\dataset\\recorded_data_gen'

def define_label(number):
    index=number-50
    raw_arr=np.zeros((90))
    raw_arr[index]=1
    return raw_arr

def read_all_data_and_labeling(folder_data):
    values=[]
    labels=[]

    for root, _dir, filenames in os.walk(folder_data):
        for filename in filenames:
            splited_filename=filename.split("_")
            label_from_filename=splited_filename[1].split(".")
            label=label_from_filename[0]
            one_hot_label=define_label(int(label))
            file_path = root + '\\' + filename
            data = rf.read_data_trainning_file(file_path)
            value=data.to_numpy()
            rows=value.shape[0]
            for j in range(rows):
                values.append(value[j, 1:])
            for i in range(rows):
                labels.append(one_hot_label)
            # list_data_frame.append(data)
    return values, labels



# data_train duoi dang dataframe
def generate_spec_batches(data_train, batch_size):
    values, labels=data_train


    data_len = len(labels)


    # shuffle data
    shuffle_seq = np.random.permutation(range(data_len))
    values = [values[idx] for idx in shuffle_seq]
    labels = [labels[idx] for idx in shuffle_seq]

    # generate batches
    num_batch = int(data_len / batch_size)
    data_batches = []
    label_batches = []
    for batch_idx in range(num_batch):
        batch_start = batch_idx * batch_size
        batch_end = np.min([(batch_idx + 1) * batch_size, data_len])
        data_batch = values[batch_start: batch_end]
        label_batch = labels[batch_start: batch_end]
        data_batches.append(data_batch)
        label_batches.append(label_batch)

    # print(data)
    return data_batches, label_batches


# file_test = "D:\\project\\img_pro_tensorflow\\dataset\\recorded_data_gen\\data_50.csv"
#
# dic_data = read_data_trainning_file(file_test, columns=['value', 'label'])
# print(dic_data.value)
data = read_all_data_and_labeling(CONFIG_DATA_FOLDER)
generate_spec_batches(data, 4)
