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

# Scale
scaler = MinMaxScaler()
logreturn_scaled = scaler.fit_transform(logreturn.reshape(-1, 1)).flatten()

# Prepare input sequences
def create_inout_sequences(data, input_window, output_window):
    X, y = [], []
    for i in range(len(data) - input_window - output_window):
        X.append(data[i:i+input_window])
        y.append(data[i+input_window:i+input_window+output_window])
    return np.array(X), np.array(y).squeeze()

# Parameters
input_window = 10
output_window = 1
X, y = create_inout_sequences(logreturn_scaled, input_window, output_window)
X = np.expand_dims(X, axis=-1)

# Define model builder
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_window, 1)))
    model.add(layers.LSTM(units=hp.Int('units', 32, 128, step=16), return_sequences=False))
    model.add(layers.Dense(output_window))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('lr', [1e-2, 1e-3, 1e-4])),
        loss='mse',
        metrics=['mae']
    )
    return model

# Cross-validation setup
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
best_val_loss = np.inf
best_model = None

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    print(f"\n Fold {fold+1}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=30,
        factor=3,
        directory='kt_dir',
        project_name=f'fold_{fold+1}',
        overwrite=True
    )

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val),
                 callbacks=[stop_early], verbose=1)

    best_hps = tuner.get_best_hyperparameters(1)[0]
    print(f" Best Hyperparameters for Fold {fold+1}:")
    print(f"Units: {best_hps.get('units')}, Learning Rate: {best_hps.get('lr')}")

    model = tuner.hypermodel.build(best_hps)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=[stop_early],
        verbose=1
    )

    # Print losses per epoch
    for epoch, (tr, vl) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
        print(f"Epoch {epoch+1}: Train Loss = {tr:.6f}, Val Loss = {vl:.6f}")

    # Plot training and validation loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Fold {fold+1} - Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
    print(f" Fold {fold+1} Final Validation Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

# Save best model (native Keras format)
best_model.save("best_model_cv.keras")
print("\n Best cross-validated model saved as `best_model_cv.keras`")


## VALIDATION 

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the trained model
model = load_model("best_model_cv.keras")

# Parameters
input_window = 10
output_window = 1

# Folder with test files
data_folder_INR = "/content/drive/MyDrive/Writing workshop data/Final data"
csv_files = glob.glob(os.path.join(data_folder_INR, "*.csv"))

# Helper functions
def create_inout_sequences(data, input_window):
    seqs, targets = [], []
    L = len(data)
    for i in range(L - input_window - output_window):
        seqs.append(data[i:i+input_window])
        targets.append(data[i+input_window:i+input_window+output_window])
    seqs = np.array(seqs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32).squeeze()
    return np.expand_dims(seqs, axis=2), targets  # shape: (samples, input_window, 1)

def cumulative_logreturn_to_voltage(cum_log_returns, start_price):
    return start_price * np.exp(cum_log_returns)

# Accumulators
mse_list, mae_list, rmse_list, r2_list = [], [], [], []
skipped_files = []

# Loop through CSVs
for csv_path in csv_files:
    filename = os.path.basename(csv_path)
    print(f"\n Processing: {filename}")

    try:
        df = pd.read_csv(csv_path)
        close = np.array(df['V'], dtype=np.float32)
        if len(close) < input_window + output_window + 10:
            raise ValueError("Not enough data")

        logreturn = np.diff(np.log(close))
        cumsum_logreturn = logreturn.cumsum()
        split = int(0.7 * len(logreturn))  # simulate test split
        initial_price = close[split]

        test_data = cumsum_logreturn[split:]
        x_test, y_test = create_inout_sequences(test_data, input_window)

        if len(x_test) == 0:
            raise ValueError("Too short after split")

        # Predict
        predictions = model.predict(x_test, verbose=0).squeeze()
        truth = y_test

        # Rescale to original voltage
        forecast_voltages = cumulative_logreturn_to_voltage(predictions, initial_price)
        truth_voltages = cumulative_logreturn_to_voltage(truth, initial_price)

        # Metrics
        mse = mean_squared_error(truth_voltages, forecast_voltages)
        mae = mean_absolute_error(truth_voltages, forecast_voltages)
        rmse = np.sqrt(mse)
        r2 = r2_score(truth_voltages, forecast_voltages)

        mse_list.append(mse)
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)

        print(f" MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}")

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(truth_voltages, label="Actual", color='red')
        plt.plot(forecast_voltages, label="Forecast", color='blue')
        plt.title(f"Forecast vs Actual: {filename}")
        plt.xlabel("Time Steps")
        plt.ylabel("Voltage")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f" Skipping {filename}: {str(e)}")
        skipped_files.append(filename)

# Summary
print("\n📊 Average Metrics Across All Valid Files:")
print(f"Avg MSE:  {np.mean(mse_list):.6f}")
print(f"Avg MAE:  {np.mean(mae_list):.6f}")
print(f"Avg RMSE: {np.mean(rmse_list):.6f}")
print(f"Avg R²:   {np.mean(r2_list):.6f}")

if skipped_files:
    print("\n Skipped Files:")
    for f in skipped_files:
        print(f"- {f}"
