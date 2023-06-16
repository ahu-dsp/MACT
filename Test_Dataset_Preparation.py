import os
import numpy as np
import pywt
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
import math
DATA_POINTS_PER_FILE = 2560#2560,32768
TIME_PER_REC = 0.1#0.1,1.28
SAMPLING_FREQ = 25600  # 25.6 KHz
SAMPLING_PERIOD = 1.0 / SAMPLING_FREQ

# WIN_SIZE = 20

WAVELET_TYPE = 'morl'

VAL_SPLIT = 0.1

np.random.seed(1234)

def load_df(pkz_file):
    with open(pkz_file, 'rb') as f:
        df=pkl.load(f)
    return df

def df_row_ind_to_data_range(ind):
    return (DATA_POINTS_PER_FILE*ind, DATA_POINTS_PER_FILE*(ind+1))


def extract_feature_image(ind, feature_name='Horizontal_vibration_signals'):
    data_range = df_row_ind_to_data_range(ind)
    data = df[feature_name].values[data_range[0]:data_range[1]]

    return data

main_dir = 'D:\论文和代码\lunwen\轴承寿命预测\RUL_Prediction-main\phm\phm\Full_Test_Set'
pkz_file=main_dir+'\\bearing1_3.pkz'
df=load_df(pkz_file)
df.head()

no_of_rows = df.shape[0]
no_of_files = int(no_of_rows / DATA_POINTS_PER_FILE)
print(no_of_rows, no_of_files)

data = {'x': [],'y': []}
for i in range(0, no_of_files):
    data_h = extract_feature_image(i, feature_name='horiz accel')
    x_ = data_h
    y_ = 1 - (i / (no_of_files - 1))
    data['x'].append(x_)
    data['y'].append(y_)
data['x'] = np.array(data['x'])
data['y'] = np.array(data['y'])

print(no_of_files, data['x'].shape)

out_file = main_dir+'\\bearing1_3_test_data_quan_1d.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(data, f)


