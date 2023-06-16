import os
import numpy as np
import pandas as pd
import pickle as pkl

def read_data_as_df(base_dir):
  '''
  saves each file in the base_dir as a df and concatenate all dfs into one
  '''
  if base_dir[-1]!='/':
    base_dir += '/'

  dfs=[]
  for f in sorted(os.listdir(base_dir)):
    print(f)
    df=pd.read_csv(base_dir+f, header=None, names=['hour', 'minute', 'second', 'microsecond', 'horiz accel', 'vert accel'])
    dfs.append(df)
  return pd.concat(dfs)

def process(base_dir, out_file):
  '''
  dumps combined dataframes into pkz (pickle) files for faster retreival
  '''
  df=read_data_as_df(base_dir)
  assert df.shape[0]==len(os.listdir(base_dir))*DATA_POINTS_PER_FILE
  with open(out_file, 'wb') as pfile:
    pkl.dump(df, pfile)
  print('{0} saved'.format(out_file))

DATA_POINTS_PER_FILE=2560
main_dir = 'D:\论文和代码\lunwen\轴承寿命预测\RUL_Prediction-main\phm\phm\Learning_set'  #Here is the address of the data set
# main_dir = 'D:\论文和代码\lunwen\轴承寿命预测\RUL_Prediction-main\phm\phm\Full_Test_Set' #Here is the address of the data set
base_dir, out_file= main_dir+'\\Bearing1_1/', main_dir+'\\bearing1_1.pkz'
process(base_dir, out_file)
base_dir, out_file= main_dir+'\\Bearing1_2/', main_dir+'\\bearing1_2.pkz'
process(base_dir, out_file)
base_dir, out_file= main_dir+'\\Bearing1_3/', main_dir+'\\bearing1_3.pkz'
process(base_dir, out_file)
base_dir, out_file= main_dir+'\\Bearing1_4/', main_dir+'\\bearing1_4.pkz'
process(base_dir, out_file)
base_dir, out_file= main_dir+'\\Bearing1_5/', main_dir+'\\bearing1_5.pkz'
process(base_dir, out_file)
base_dir, out_file= main_dir+'\\Bearing1_6/', main_dir+'\\bearing1_6.pkz'
process(base_dir, out_file)
base_dir, out_file= main_dir+'\\Bearing1_7/', main_dir+'\\bearing2_7.pkz'
process(base_dir, out_file)
