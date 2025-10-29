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
