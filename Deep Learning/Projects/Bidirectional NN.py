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

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# reshape into X=t and Y=t+1
look_back = 12
X_train, Y_train = create_dataset(df_train, look_back)
X_valid, Y_valid = create_dataset(df_valid, look_back)

X_train.shape
Y_valid.shape

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_valid = np.reshape(X_valid, (X_valid.shape[0], 1, X_valid.shape[1]))

from tensorflow.keras.regularizers import L1L2
import tensorflow as tf
from keras.regularizers import l2

units=50
model = Sequential()
model.add(Bidirectional(LSTM(units, kernel_regularizer=l2(0.0001),return_sequences=True),
                             input_shape=(X_train.shape[1], look_back), 
                             ))
model.add(Bidirectional(LSTM(units)))
model.add(Dropout(0.2)) #0.1 or 0.2
model.add(Dense(1))
model.add(Activation('relu'))

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(loss='mean_squared_error', optimizer=opt, metrics='MeanSquaredError')  

history = model.fit(X_train, Y_train, epochs=500, batch_size=200, validation_data=(X_valid, Y_valid), 
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)
# Training Phase
model.summary()

#%%
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();

plt.figure(figsize=(8,4))
plt.plot(history.history['mean_squared_error'], label='MSE')
plt.plot(history.history['val_mean_squared_error'], label='Val MSE')
plt.title('model metrics')
plt.ylabel('MSE')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();

#%%
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# make predictions
train_predict = model.predict(X_train)
valid_predict = model.predict(X_valid)

# invert predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
valid_predict = scaler.inverse_transform(valid_predict)
Y_valid = scaler.inverse_transform([Y_valid])

print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('Valid Mean Absolute Error:', mean_absolute_error(Y_valid[0], valid_predict[:,0]))
print('Valid Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_valid[0], valid_predict[:,0])))

print('Coefficient of determination: %.4f R2' % r2_score(Y_valid.T, valid_predict))

#%%
# shift train predictions for plotting
trainPredictPlot = np.empty_like(df_train)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

