
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
train_path = "Introduce your training path here"
val_path = "Introduce your validation path here"
test_path ="Introduce your testing path here"

df_train = pd.read_csv(train_path, index_col=(6))
df_val = pd.read_csv(val_path, index_col=(6))
df_test = pd.read_csv(test_path, index_col=(6))

X_train, y_train, _, f_scaler, t_scaler = transform_cm(df_train, fit_scaler=True)
X_val, y_val, _, _, _ = transform_cm(df_val, f_scaler, t_scaler)
X_test, y_test, _, _, _ = transform_cm(df_test, f_scaler, t_scaler)

# -------------------------------------
# MODEL SETUP (Keras)
# -------------------------------------
num_ann_units = 64#64
learning_rate = 1e-4
gamma = 0.95
weight_decay = 1e-4


