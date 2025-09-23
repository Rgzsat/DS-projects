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
    print(f"\n🔁 Fold {fold+1}")

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
    print(f"🔍 Best Hyperparameters for Fold {fold+1}:")
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
    print(f"✅ Fold {fold+1} Final Validation Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

# Save best model (native Keras format)
best_model.save("best_model_cv.keras")
print("\n🎉 Best cross-validated model saved as `best_model_cv.keras`")
