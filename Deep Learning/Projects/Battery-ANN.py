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
# To transform the data, considering specific features based on user criteria
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

# FILES & TRANSFORM
# -------------------------------------
train_path = "Insert your training path here"
val_path = "Insert your validation path here"
test_path = "Insert your testing path here"

# ---------------------------------------
# LOAD AND TRANSFORM DATA
# ---------------------------------------
df_train = pd.read_csv(train_path, index_col=(6))
df_val = pd.read_csv(val_path, index_col=(6))
df_test = pd.read_csv(test_path, index_col=(6))

X_train, y_train, _, f_scaler, t_scaler = transform_cm(df_train, fit_scaler=True)
X_val, y_val, _, _, _ = transform_cm(df_val, f_scaler, t_scaler)
X_test, y_test, _, _, _ = transform_cm(df_test, f_scaler, t_scaler)

# Convert to tensors
X_train_tensor = torch.tensor(X_train).to(device)
y_train_tensor = torch.tensor(y_train).to(device)
X_val_tensor = torch.tensor(X_val).to(device)
y_val_tensor = torch.tensor(y_val).to(device)
X_test_tensor = torch.tensor(X_test).to(device)
y_test_tensor = torch.tensor(y_test).to(device)

# Dataloaders
bs = 128
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=bs, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=bs, shuffle=False)

# ---------------------------------------
# MODEL DEFINITION
# ---------------------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

model = MLPRegressor(input_dim=X_train.shape[1]).to(device)

# ---------------------------------------
# TRAINING SETUP
# ---------------------------------------
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
loss_func = nn.MSELoss()

n_epochs = 50
patience_limit = 10
patience = patience_limit
train_loss_all = []
val_loss_all = []
best_loss = float('inf')
best_model_weights = copy.deepcopy(model.state_dict())
