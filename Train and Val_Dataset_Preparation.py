import os
import numpy as np
import pywt
import pandas as pd
import pickle as pkl
import torch
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
    # use window to process(= prepare, develop) 1D signal
    # data = np.array([np.mean(data[i:i+WIN_SIZE]) for i in range(0, DATA_POINTS_PER_FILE, WIN_SIZE)])

    return data

main_dir = 'D:\论文和代码\lunwen\轴承寿命预测\RUL_Prediction-main\phm\phm\Learning_set'


pkz_file = main_dir + '\\bearing1_1.pkz'
df = load_df(pkz_file)
print(df.head())

no_of_rows = df.shape[0]
no_of_files = int(no_of_rows / DATA_POINTS_PER_FILE)
print(no_of_rows, no_of_files)

data = {'x': [], 'y': []}
for i in range(0, no_of_files):
    data_h = extract_feature_image(i, feature_name='horiz accel')
    x_ = data_h
    y_ = 1-(i/(no_of_files-1))
    data['x'].append(x_)
    data['y'].append(y_)
data['x']=np.array(data['x'])
data['y']=np.array(data['y'])

print(no_of_files, data['x'].shape, data['y'])
print(data['x'])

no_of_val = int(VAL_SPLIT*no_of_files)
perm = np.random.permutation(no_of_files)
val_data = {'x': data['x'][perm[0:no_of_val]], 'y': data['y'][perm[0:no_of_val]]}
train_data = {'x': data['x'][perm[no_of_val: ]], 'y': data['y'][perm[no_of_val: ]]}


print(no_of_val, val_data['x'].shape, val_data['y'].shape)
print(no_of_files-no_of_val, train_data['x'].shape, train_data['y'].shape)

out_file = main_dir + '\\bearing1_1_val_data_origin.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(val_data, f)


out_file = main_dir + '\\bearing1_1_train_data_origin.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(train_data, f)


pkz_file = main_dir + '\\bearing1_2.pkz'
df = load_df(pkz_file)
print(df.head())

no_of_rows = df.shape[0]
no_of_files = int(no_of_rows / DATA_POINTS_PER_FILE)
print(no_of_rows, no_of_files)

data = {'x': [], 'y': []}
for i in range(0, no_of_files):
    data_h = extract_feature_image(i, feature_name='horiz accel')
    x_ = data_h
    y_ = 1-(i/(no_of_files-1))
    data['x'].append(x_)
    data['y'].append(y_)
data['x']=np.array(data['x'])
data['y']=np.array(data['y'])

print(no_of_files, data['x'].shape, data['y'])
print(data['x'])

no_of_val = int(VAL_SPLIT*no_of_files)
perm = np.random.permutation(no_of_files)
val_data = {'x': data['x'][perm[0:no_of_val]], 'y': data['y'][perm[0:no_of_val]]}
train_data = {'x': data['x'][perm[no_of_val: ]], 'y': data['y'][perm[no_of_val: ]]}


print(no_of_val, val_data['x'].shape, val_data['y'].shape)
print(no_of_files-no_of_val, train_data['x'].shape, train_data['y'].shape)

out_file = main_dir + '\\bearing1_2_val_data_origin.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(val_data, f)


out_file = main_dir + '\\bearing1_2_train_data_origin.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(train_data, f)


pkz_file = main_dir + '\\bearing1_3.pkz'
df = load_df(pkz_file)
print(df.head())

no_of_rows = df.shape[0]
no_of_files = int(no_of_rows / DATA_POINTS_PER_FILE)
print(no_of_rows, no_of_files)

data = {'x': [], 'y': []}
for i in range(0, no_of_files):
    data_h = extract_feature_image(i, feature_name='horiz accel')
    x_ = data_h
    y_ = 1-(i/(no_of_files-1))
    data['x'].append(x_)
    data['y'].append(y_)
data['x']=np.array(data['x'])
data['y']=np.array(data['y'])

print(no_of_files, data['x'].shape, data['y'])
print(data['x'])

no_of_val = int(VAL_SPLIT*no_of_files)
perm = np.random.permutation(no_of_files)
val_data = {'x': data['x'][perm[0:no_of_val]], 'y': data['y'][perm[0:no_of_val]]}
train_data = {'x': data['x'][perm[no_of_val: ]], 'y': data['y'][perm[no_of_val: ]]}


print(no_of_val, val_data['x'].shape, val_data['y'].shape)
print(no_of_files-no_of_val, train_data['x'].shape, train_data['y'].shape)

out_file = main_dir + '\\bearing1_3_val_data_origin.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(val_data, f)


out_file = main_dir + '\\bearing1_3_train_data_origin.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(train_data, f)



pkz_file = main_dir + '\\bearing1_4.pkz'
df = load_df(pkz_file)
print(df.head())

no_of_rows = df.shape[0]
no_of_files = int(no_of_rows / DATA_POINTS_PER_FILE)
print(no_of_rows, no_of_files)

data = {'x': [], 'y': []}
for i in range(0, no_of_files):
    data_h = extract_feature_image(i, feature_name='horiz accel')
    x_ = data_h
    y_ = 1-(i/(no_of_files-1))
    data['x'].append(x_)
    data['y'].append(y_)
data['x']=np.array(data['x'])
data['y']=np.array(data['y'])

print(no_of_files, data['x'].shape, data['y'])
print(data['x'])

no_of_val = int(VAL_SPLIT*no_of_files)
perm = np.random.permutation(no_of_files)
val_data = {'x': data['x'][perm[0:no_of_val]], 'y': data['y'][perm[0:no_of_val]]}
train_data = {'x': data['x'][perm[no_of_val: ]], 'y': data['y'][perm[no_of_val: ]]}


print(no_of_val, val_data['x'].shape, val_data['y'].shape)
print(no_of_files-no_of_val, train_data['x'].shape, train_data['y'].shape)

out_file = main_dir + '\\bearing1_4_val_data_origin.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(val_data, f)


out_file = main_dir + '\\bearing1_4_train_data_origin.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(train_data, f)



pkz_file = main_dir + '\\bearing1_5.pkz'
df = load_df(pkz_file)
print(df.head())

no_of_rows = df.shape[0]
no_of_files = int(no_of_rows / DATA_POINTS_PER_FILE)
print(no_of_rows, no_of_files)

data = {'x': [], 'y': []}
for i in range(0, no_of_files):
    data_h = extract_feature_image(i, feature_name='horiz accel')
    x_ = data_h
    y_ = 1-(i/(no_of_files-1))
    data['x'].append(x_)
    data['y'].append(y_)
data['x']=np.array(data['x'])
data['y']=np.array(data['y'])

print(no_of_files, data['x'].shape, data['y'])
print(data['x'])

no_of_val = int(VAL_SPLIT*no_of_files)
perm = np.random.permutation(no_of_files)
val_data = {'x': data['x'][perm[0:no_of_val]], 'y': data['y'][perm[0:no_of_val]]}
train_data = {'x': data['x'][perm[no_of_val: ]], 'y': data['y'][perm[no_of_val: ]]}


print(no_of_val, val_data['x'].shape, val_data['y'].shape)
print(no_of_files-no_of_val, train_data['x'].shape, train_data['y'].shape)

out_file = main_dir + '\\bearing1_5_val_data_origin.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(val_data, f)


out_file = main_dir + '\\bearing1_5_train_data_origin.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(train_data, f)



pkz_file = main_dir + '\\bearing1_6.pkz'
df = load_df(pkz_file)
print(df.head())

no_of_rows = df.shape[0]
no_of_files = int(no_of_rows / DATA_POINTS_PER_FILE)
print(no_of_rows, no_of_files)

data = {'x': [], 'y': []}
for i in range(0, no_of_files):
    data_h = extract_feature_image(i, feature_name='horiz accel')
    x_ = data_h
    y_ = 1-(i/(no_of_files-1))
    data['x'].append(x_)
    data['y'].append(y_)
data['x']=np.array(data['x'])
data['y']=np.array(data['y'])

print(no_of_files, data['x'].shape, data['y'])
print(data['x'])

no_of_val = int(VAL_SPLIT*no_of_files)
perm = np.random.permutation(no_of_files)
val_data = {'x': data['x'][perm[0:no_of_val]], 'y': data['y'][perm[0:no_of_val]]}
train_data = {'x': data['x'][perm[no_of_val: ]], 'y': data['y'][perm[no_of_val: ]]}


print(no_of_val, val_data['x'].shape, val_data['y'].shape)
print(no_of_files-no_of_val, train_data['x'].shape, train_data['y'].shape)

out_file = main_dir + '\\bearing1_6_val_data_origin.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(val_data, f)


out_file = main_dir + '\\bearing1_6_train_data_origin.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(train_data, f)



pkz_file = main_dir + '\\bearing1_7.pkz'
df = load_df(pkz_file)
print(df.head())

no_of_rows = df.shape[0]
no_of_files = int(no_of_rows / DATA_POINTS_PER_FILE)
print(no_of_rows, no_of_files)

data = {'x': [], 'y': []}
for i in range(0, no_of_files):
    data_h = extract_feature_image(i, feature_name='horiz accel')
    x_ = data_h
    y_ = 1-(i/(no_of_files-1))
    data['x'].append(x_)
    data['y'].append(y_)
data['x']=np.array(data['x'])
data['y']=np.array(data['y'])

print(no_of_files, data['x'].shape, data['y'])
print(data['x'])

no_of_val = int(VAL_SPLIT*no_of_files)
perm = np.random.permutation(no_of_files)
val_data = {'x': data['x'][perm[0:no_of_val]], 'y': data['y'][perm[0:no_of_val]]}
train_data = {'x': data['x'][perm[no_of_val: ]], 'y': data['y'][perm[no_of_val: ]]}


print(no_of_val, val_data['x'].shape, val_data['y'].shape)
print(no_of_files-no_of_val, train_data['x'].shape, train_data['y'].shape)

out_file = main_dir + '\\bearing1_7_val_data_origin.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(val_data, f)


out_file = main_dir + '\\bearing1_7_train_data_origin.pkz'
with open(out_file, 'wb') as f:
    pkl.dump(train_data, f)