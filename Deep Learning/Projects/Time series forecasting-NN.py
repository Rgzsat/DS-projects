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

history = model.fit(X_train, Y_train, epochs=500, batch_size=90, validation_data=(X_valid, Y_valid), 
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

#%%
plt.figure(figsize=(8,4))
plt.plot(history.history['mean_squared_error'], label='MSE')
plt.plot(history.history['val_mean_squared_error'], label='Val MSE')
plt.title('model metrics')
plt.ylabel('MSE')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();

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

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting
validPredictPlot = np.empty_like(dataset)
validPredictPlot[:, :] = np.nan
validPredictPlot[len(train_predict)+(look_back*2)+1:len(dataset)-1, :] = valid_predict


#%%
import seaborn as sns

actual= (np.r_[Y_train.T, Y_valid.T])
pred= (np.r_[train_predict, valid_predict])

#plt.plot(scaler.inverse_transform(dataset), label= 'actual')
plt.plot(actual, label= 'actual')
plt.plot(pred, label= 'predicted', color= 'orange')
#plt.plot(trainPredictPlot, color= 'orange', label= 'predicted')
#plt.plot(validPredictPlot, color= 'orange')
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.xlabel('Time [sec]')
plt.ylabel('Voltage [V]')
plt.title('LSTM implementation')
plt.legend()
plt.show()

dataset_test= pd.read_csv(r'C:\Users\rogilb\Downloads\Testing cells\35A-1-A-4.csv',index_col= 6)#[:-1]
dataset_test= np.array(dataset_test['V'])
dataset_test = np.reshape(dataset_test, (-1, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
df_test = scaler2.fit_transform(dataset_test)

X_test, Y_test = create_dataset(df_test, look_back)

X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

test_predict = model.predict(X_test)

test_predict = scaler2.inverse_transform(test_predict)
Y_test = scaler2.inverse_transform([Y_test])

print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))

