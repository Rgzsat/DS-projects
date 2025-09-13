import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import matplotlib.pyplot as plt

# Load your training data (log returns)
df = pd.read_csv(data_path)  # Replace with your file
close = np.array(df['V'], dtype=np.float32)
logreturn = np.diff(np.log(close)).astype(np.float32)

