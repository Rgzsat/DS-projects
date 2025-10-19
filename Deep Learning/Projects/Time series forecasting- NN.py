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
