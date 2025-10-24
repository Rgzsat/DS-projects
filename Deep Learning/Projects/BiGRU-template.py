
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
