import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LeakyReLU
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.layers import Activation, Dense



import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
from math import sqrt

#%%
dataset_train= pd.read_csv(r'C:\Users\rogilb\Downloads\Testing cells\1A-1-B-1.csv',index_col= 6)#[:-1]
dataset_valid= pd.read_csv(r'C:\Users\rogilb\Downloads\Testing cells\10A-1-A-1.csv',index_col= 6)#[:-1]


scaler = MinMaxScaler(feature_range=(0, 1))

def df(dataset):
    #v= dataset['V']
    #dataset= np.array(v.iloc[::-1])
    dataset=  np.array(dataset['V'])
    dataset = np.reshape(dataset, (-1, 1))
    dataset = scaler.fit_transform(dataset)
    
    return dataset

df_train= df(dataset_train)
df_valid= df(dataset_valid)

train_size = (len(df_train))
val_size = len(df_valid)
