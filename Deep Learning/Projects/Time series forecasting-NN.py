import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

dataset_train= #Introduce here your path for the training dataset

look_back = 8#

dataset= np.array(dataset_train['V'])
dataset = np.reshape(dataset, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

#df_train= f_reshape(dataset_train)
df_train= dataset
train_size = int(len(df_train) * 0.8)
val_size = len(df_train) - train_size
train, valid = df_train[0:train_size,:], df_train[train_size:len(dataset),:]   

def create_dataset(dataset, look_back=look_back):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# reshape into X=t and Y=t+1
X_train, Y_train = create_dataset(train, look_back)
X_valid, Y_valid = create_dataset(valid, look_back)

X_train.shape
Y_valid.shape

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_valid = np.reshape(X_valid, (X_valid.shape[0], 1, X_valid.shape[1]))

from tensorflow.keras.regularizers import L1L2
import tensorflow as tf
from keras.regularizers import l2

model = Sequential()

model.add(tf.keras.layers.LSTM(200, activation='relu', kernel_regularizer =l2(0.0001)))
model.add(tf.keras.layers.InputLayer(input_shape=(1, look_back)))
model.add(Dropout(0.2)) #0.1 or 0.2
model.add(Dense(1))

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.AdamW(learning_rate=lr_schedule)
