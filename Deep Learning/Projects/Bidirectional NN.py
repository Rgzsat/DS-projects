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
dataset_train= "Introduce here your training dataset or path"
dataset_valid= "Introduce here your validation dataset or path"

scaler = MinMaxScaler(feature_range=(0, 1))

def df(dataset):
    dataset=  np.array(dataset['V'])
    dataset = np.reshape(dataset, (-1, 1))
    dataset = scaler.fit_transform(dataset)
    
    return dataset

df_train= df(dataset_train)
df_valid= df(dataset_valid)

train_size = (len(df_train))
val_size = len(df_valid)


