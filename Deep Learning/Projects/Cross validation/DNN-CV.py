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

dataset_train = "TRAINING PATH"
dataset_valid= "VALIDATION PATH"

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

X_train, y_train= df(dataset_train)[0], df(dataset_train)[1]
X_valid, y_valid= df(dataset_valid)[0], df(dataset_valid)[1]

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

#%% KFOLD

from sklearn.model_selection import KFold 
from keras.regularizers import l2

#choose how many folds/subsets you want
num_folds = 10

# Merge inputs and targets
inputs = np.concatenate((X_train, X_valid), axis=0)
targets = np.concatenate((y_train, y_valid), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# Define per-fold score containers
mse_per_fold = []
mae_per_fold = []
loss_per_fold = []

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

  # Define the model architecture
  model = Sequential([
      Dense(64, input_shape=(X_train.shape[1],), activation='relu', 
            kernel_regularizer=l2(0.001)),
      Dense(64, activation='relu'),
      Dense(64, activation='relu'),
      Dense(4, activation='softmax')
  ])
  lr_schedule = keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=1e-4,
      decay_steps=10000,
      decay_rate=0.9)
  opt = keras.optimizers.Adam(learning_rate=lr_schedule)

  model.add(Dense(units=50, kernel_initializer='normal', activation='relu'))
  model.add(Dropout(0.2)) #0.1 or 0.2
  model.add(Dense(1))
  
  model.compile(optimizer=opt, loss='mse', metrics=['mse', 'mae'])


  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(X_train, y_train, epochs=500, batch_size=250, validation_data=(X_valid, y_valid), 
                      callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)


  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}; {model.metrics_names[2]} of {scores[2]}')
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
print('Average scores for all folds ANN-MLP:')
print(f'> MSE: {np.mean(mse_per_fold)} (+- {np.std(mse_per_fold)})')
print(f'> MAE: {np.mean(mae_per_fold)} (+- {np.std(mae_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

#%%

model.save('MLP.h5')
