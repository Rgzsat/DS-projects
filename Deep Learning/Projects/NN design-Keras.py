
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

# -------------------------------------
# TRANSFORM FUNCTION
# -------------------------------------

def transform_cm(df, feature_scaler=None, target_scaler=None, fit_scaler=False):
    df = df.copy()

    cap = df['mAh'] / 1000.0
    max_cap = cap.max() if cap.max() != 0 else 1e-6
    dod = (max_cap - cap) / max_cap
    soc = 1 - dod
    df['SOC'] = soc

    y = df['V'].values.reshape(-1, 1)
    X = df[['mAh', 'SOC']].values

    if feature_scaler is None:
        feature_scaler = MinMaxScaler()
    if target_scaler is None:
        target_scaler = MinMaxScaler()

    if fit_scaler:
        X_scaled = feature_scaler.fit_transform(X)
        y_scaled = target_scaler.fit_transform(y)
    else:
        X_scaled = feature_scaler.transform(X)
        y_scaled = target_scaler.transform(y)

    return X_scaled.astype(np.float32), y_scaled.astype(np.float32), df, feature_scaler, target_scaler

# -------------------------------------
# FILES & TRANSFORM
# -------------------------------------
train_path = r'C:\Users\47406\Downloads\Final article\RIGOL\1B1-1A.csv'
val_path = r'C:\Users\47406\Downloads\Final article\RIGOL\1B1-3A.csv'
test_path = r'C:\Users\47406\Downloads\Coding 2023\Coding-initial datasets\2A4-10A.csv'

