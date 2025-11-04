
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers import Dense,Flatten
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import  Dropout, Flatten

import plotly.offline as py #visualization
py.init_notebook_mode(connected=True) #visualization
from sklearn.preprocessing import MinMaxScaler

dataset_train = "TRAINING PATH"
dataset_valid= "VALIDATION PATH"
scaler = MinMaxScaler(feature_range=(0, 1))


def df(dataset):
    dataset=(dataset.drop('time', axis=1))
    X = dataset.drop('V', axis=1)
    X= scaler.fit_transform(X)
    
    y = np.array(dataset.V)
    y = np.reshape(y, (-1, 1))
    y= scaler.fit_transform(y)
    
    return X, y

X_train, y_train= df(dataset_train)[0], df(dataset_train)[1]
X_valid, y_valid= df(dataset_valid)[0], df(dataset_valid)[1]

X_train_series = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_valid_series = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
print('Train set shape', X_train_series.shape)
print('Validation set shape', X_valid_series.shape)

#%% KFOLD

from sklearn.model_selection import KFold 
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers

epochs = 500

#choose how many folds/subsets you want
num_folds = 10

# Merge inputs and targets
inputs = np.concatenate((X_train_series, X_valid_series), axis=0)
targets = np.concatenate((y_train, y_valid), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# Define per-fold score containers
mse_per_fold = []
mae_per_fold = []
loss_per_fold = []
lr = 0.00001
#adam = tf.keras.optimizers.Adam(lr)
adam = tf.keras.optimizers.legacy.Adam(lr)

# K-fold Cross Validation model evaluation
fold_no 

for train, test in kfold.split(inputs, targets):

  # Define the model architecture
  model_cnn = tf.keras.models.Sequential()
  model_cnn.add(tf.keras.layers.Conv1D(filters=300, kernel_size=2, activation='relu', 
                       kernel_regularizer =l2(0.01), input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
  #model_cnn.add(MaxPooling1D(pool_size=2))
  model_cnn.add(tf.keras.layers.Flatten())
  
  model_cnn.add(Dense(50, activation='relu')) #50
  
  model_cnn.add(tf.keras.layers.Dense(1))

  model_cnn.add(Dropout(0.2))
  
  lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9)
  opt = keras.optimizers.Adam(learning_rate=lr_schedule)


  model_cnn.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])


  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model_cnn.fit(X_train_series, y_train, epochs=epochs, batch_size= 200,
                              validation_data=(X_valid_series, y_valid), 
                      callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

  # Generate generalization metrics
  scores = model_cnn.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model_cnn.metrics_names[0]} of {scores[0]}; {model_cnn.metrics_names[1]} of {scores[1]}; {model_cnn.metrics_names[2]} of {scores[2]}')
  mse_per_fold.append(scores[1])
  mae_per_fold.append(scores[2])
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(mse_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - MSE: {mse_per_fold[i]} - MAE: {mae_per_fold[i]}')
print('------------------------------------------------------------------------')
print('Average scores for all folds CNN dense:')
print(f'> MSE: {np.mean(mse_per_fold)} (+- {np.std(mse_per_fold)})')
print(f'> MAE: {np.mean(mae_per_fold)} (+- {np.std(mae_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
