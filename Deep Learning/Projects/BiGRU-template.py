
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras.layers import  Dropout, Flatten

from keras.layers import Activation, Dense

import skopt
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Integer 
from skopt.plots import plot_convergence, plot_objective, plot_evaluations, plot_histogram, plot_objective_2D 
from skopt.utils import use_named_args

dataset_train "Introduce your training data path"
dataset_valid= "Introduce your validation data path"
scaler = MinMaxScaler(feature_range=(0, 1))


def df(dataset):
    cap= dataset['mAh']/1000
    max_cap=cap[len(cap)-1]
    dod= (max_cap - cap)/max_cap
    dataset['SOC']= dod
    X= dataset
    X= scaler.fit_transform(X)
    
    ir= (4.1-dataset['V'])/dataset['I']
    yi= ir[0]*dataset['I']+dataset['V']
    y = np.array(yi)
    y = np.reshape(y, (-1, 1))
    y= scaler.fit_transform(y)
  
    return X, y

train_size = int(len(dataset_train) * 0.7)
val_size = len(dataset_train) - train_size


x_train, y_train= df(dataset_train)[0], df(dataset_train)[1]
x_valid, y_valid= df(dataset_valid)[0], df(dataset_valid)[1]

X_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
X_valid = x_valid.reshape((x_valid.shape[0], 1, x_valid.shape[1]))

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

from tensorflow.keras.regularizers import L1L2
import tensorflow as tf
from keras.regularizers import l2


units=250 #200, 300
model = Sequential()
model.add(Bidirectional(GRU(units, kernel_regularizer=l2(0.0001),return_sequences=True),
                             input_shape=(X_train.shape[1], X_train.shape[2]), 
                             ))


model.add(Bidirectional(GRU(units)))
model.add(Dropout(0.2)) #0.1 or 0.2
model.add(Dense(1))
model.add(Activation('relu'))

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(loss='mean_squared_error', optimizer=opt, metrics='MeanSquaredError')     

history = model.fit(X_train, y_train, epochs=500, batch_size=200, validation_data=(X_valid, y_valid), 
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
#from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

#%%

import seaborn as sns

plt.figure(1)
plt.plot((y_train), label= 'actual')
plt.plot((train_predict), label= 'predicted')
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.xlabel('Time [sec]')
plt.ylabel('Voltage [V]')
plt.title('BiGRU implementation-Training')
plt.legend()
plt.show()

# make predictions
train_predict = model.predict(X_train)
valid_predict = model.predict(X_valid)

# invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train)
valid_predict = scaler.inverse_transform(valid_predict)
y_valid = scaler.inverse_transform(y_valid)

print(valid_predict.shape)

print(valid_predict)

print('Train Mean Absolute Error:', mean_absolute_error(y_train, train_predict))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(y_train, train_predict)))
print('Valid Mean Absolute Error:', mean_absolute_error(y_valid, valid_predict))
print('Valid Root Mean Squared Error:',np.sqrt(mean_squared_error(y_valid, valid_predict)))

print(' Training Coefficient of determination: %.4f R2' % r2_score(y_train, train_predict))
print(' Validation Coefficient of determination: %.4f R2' % r2_score(y_valid, valid_predict))

#%%
plt.figure(2)
plt.plot((y_valid), label= 'actual')
#plt.plot(sorted(valid_predict, reverse= True), label= 'predicted', color= 'red')
plt.plot((valid_predict), label= 'predicted')
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.xlabel('Time [sec]')
plt.ylabel('Voltage [V]')
plt.title('BiGRU implementation-Validation')
plt.legend()
plt.show()

#%%
dataset_test= pd.read_csv(r'C:\Users\47406\Downloads\Coding-initial\2A4-10A.csv', index_col= 6)#[:-1]
x_test, y_test= df(dataset_test)[0], df(dataset_test)[1]
X_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

