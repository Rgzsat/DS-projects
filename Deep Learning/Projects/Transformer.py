import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt

# Configuration
input_window = 10
output_window = 1
batch_size = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path ="/content/drive/MyDrive/Writing workshop data/Final data/Cell4-005C-080425.csv" #LOAD YOUR PATH HERE


# Load Data
df = pd.read_csv(data_path)
close = np.array(df['V'])
logreturn = np.diff(np.log(close))
logreturn.astype(np.float32)
csum_logreturn = logreturn.cumsum()

# Visualization
fig, axs = plt.subplots(2, 1)
axs[0].plot(close, color='red')
axs[0].set_title('Voltage')
axs[1].plot(csum_logreturn, color='green')
axs[1].set_title('Cumulative Sum of Log Voltage')
fig.tight_layout()
plt.show()

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

# Transformer Model
class TransAm(nn.Module):
    def __init__(self, feature_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.input_projection = nn.Linear(1, feature_size)

    def forward(self, src):
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

# Sequence Creator
def create_inout_sequences(data, input_window):
    seqs = []
    L = len(data)
    for i in range(L - input_window - output_window):
        input_seq = data[i:i+input_window]
        target_seq = data[i+input_window:i+input_window+output_window]
        seqs.append((
    np.array(input_seq, dtype=np.float32),
    np.array(target_seq, dtype=np.float32)
))
    return seqs

# Data Loader
def get_data(data, split_ratio):
    split = int(split_ratio * len(data))
    train = 2 * data[:split].cumsum()
    test = data[split:].cumsum()

    train_seq = create_inout_sequences(train, input_window)
    test_seq = create_inout_sequences(test, input_window)

    return train_seq, test_seq

def get_batch(data, i, batch_size):
    batch = data[i:i+batch_size]
    inputs = torch.stack([torch.tensor(x[0], dtype=torch.float32) for x in batch])  # [B, seq_len]
    targets = torch.stack([torch.tensor(x[1], dtype=torch.float32) for x in batch])
    inputs = inputs.unsqueeze(-1).transpose(0, 1).to(device)  # [seq_len, B, 1]
    targets = targets.unsqueeze(-1).transpose(0, 1).to(device)
    return inputs, targets

# Training
def train(model, data, optimizer, criterion):
    model.train()
    total_loss = 0
    for i in range(0, len(data) - batch_size, batch_size):
        inputs, targets = get_batch(data, i, batch_size)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output[-output_window:], targets[-output_window:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.6)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data)

def evaluate(model, data, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, len(data) - batch_size, batch_size):
            inputs, targets = get_batch(data, i, batch_size)
            output = model(inputs)
            loss = criterion(output[-output_window:], targets[-output_window:])
            total_loss += loss.item()
    return total_loss / len(data)

def forecast_seq(model, sequences):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for i in range(0, len(sequences) - batch_size, batch_size):
            inputs, targets = get_batch(sequences, i, batch_size)
            output = model(inputs)
            predictions.append(output[-1].squeeze().cpu())
            actuals.append(targets[-1].squeeze().cpu())
    return torch.cat(predictions), torch.cat(actuals)

# Prepare Data
train_data, val_data = get_data(logreturn, 0.6)
model = TransAm().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# Training Loop
epochs = 150
best_val_loss = float("inf")
for epoch in range(1, epochs + 1):
    start = time.time()
    train_loss = train(model, train_data, optimizer, criterion)
    val_loss = evaluate(model, val_data, criterion)
    print(f"Epoch {epoch} | Time {time.time()-start:.2f}s | Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
    scheduler.step()

# Forecast & Plot
model.load_state_dict(torch.load('best_model.pt'))
forecast, truth = forecast_seq(model, val_data)
plt.plot(truth.numpy(), color='red', alpha=0.7)
plt.plot(forecast.numpy(), color='blue', linewidth=0.7)
plt.title('Actual vs Forecast')
plt.legend(['Actual', 'Forecast'])
plt.xlabel('Time Steps')
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Rescale predictions and ground truth to original voltage scale
# Step 1: Convert predictions and truth from tensors to numpy
forecast_np = forecast.numpy()
truth_np = truth.numpy()

# Step 2: Undo cumulative sum to get log returns
# We're comparing cumulative log return windows to the true ones, so reconstruct the full sequence from the last known value
initial_index = int(len(close) * 0.6)
initial_voltage = close[initial_index]

# Convert log returns back to voltage
# Proper conversion from cumulative log returns to voltage
def cumulative_logreturn_to_voltage(cum_log_returns, start_voltage):
    return start_voltage * np.exp(cum_log_returns)

forecast_voltage = cumulative_logreturn_to_voltage(forecast_np, initial_voltage)
truth_voltage = cumulative_logreturn_to_voltage(truth_np, initial_voltage)

# Metrics
mse = mean_squared_error(truth_voltage, forecast_voltage)
mae = mean_absolute_error(truth_voltage, forecast_voltage)
r2 = r2_score(truth_voltage, forecast_voltage)

print("\nPerformance Metrics on Test Data (Original Scale):")
print(f"Mean Squared Error (MSE):     {mse:.6f}")
print(f"Mean Absolute Error (MAE):    {mae:.6f}")
print(f"R-squared (R²):               {r2:.6f}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(truth_voltage, label='Actual Voltage', color='red', alpha=0.7)
plt.plot(forecast_voltage, label='Forecasted Voltage', color='blue', linewidth=0.7)
plt.title('Forecast vs Actual Voltage (Original Scale)')
plt.xlabel('Time Steps')
plt.ylabel('Voltage')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Load model
model = TransAm().to(device)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Parameters
input_window = 10
output_window = 1

# Folder containing test CSVs
data_folder_test = #Read the folder with your testing data 
csv_files = glob.glob(os.path.join(data_folder_test, "*.csv"))


# Utility Functions
def create_inout_sequences(data, input_window):
    seqs = []
    L = len(data)
    for i in range(L - input_window - output_window):
        input_seq = np.array(data[i:i+input_window], dtype=np.float32)
        target_seq = np.array(data[i+input_window:i+input_window+output_window], dtype=np.float32)
        seqs.append((input_seq, target_seq))
    return seqs

def get_batch(data, i, batch_size):
    batch = data[i:i+batch_size]
    inputs = torch.stack([torch.tensor(x[0], dtype=torch.float32) for x in batch])
    targets = torch.stack([torch.tensor(x[1], dtype=torch.float32) for x in batch])
    inputs = inputs.unsqueeze(-1).transpose(0, 1).to(device)
    targets = targets.unsqueeze(-1).transpose(0, 1).to(device)
    return inputs, targets

def forecast_seq(model, sequences):
    predictions = []
    actuals = []
    with torch.no_grad():
        for i in range(0, len(sequences) - 10, 10):
            inputs, targets = get_batch(sequences, i, 10)
            output = model(inputs)
            predictions.append(output[-1].squeeze().cpu())
            actuals.append(targets[-1].squeeze().cpu())
    return torch.cat(predictions), torch.cat(actuals)

def cumulative_logreturn_to_voltage(cum_log_returns, start_price):
    return start_price * np.exp(cum_log_returns)

# Initialize accumulators
mse_list = []
mae_list = []
rmse_list = []
r2_list = []

# Loop through each CSV
for csv_path in csv_files:
    filename = os.path.basename(csv_path)
    print(f"\n Processing: {filename}")

    # Load and preprocess
    df = pd.read_csv(csv_path)
    close = np.array(df['V'])
    logreturn = np.diff(np.log(close)).astype(np.float32)
    cumsum_logreturn = logreturn.cumsum()
    split = int(0.6 * len(logreturn))
    initial_price = close[split]

    test_data = cumsum_logreturn[split:]
    test_seq = create_inout_sequences(test_data, input_window)

    # Forecast
    forecast, truth = forecast_seq(model, test_seq)

    # Rescale to original
    forecast_np = forecast.numpy()
    truth_np = truth.numpy()
    forecast_voltages = cumulative_logreturn_to_voltage(forecast_np, initial_price)
    truth_voltages = cumulative_logreturn_to_voltage(truth_np, initial_price)

    # Metrics
    mse = mean_squared_error(truth_voltages, forecast_voltages)
    mae = mean_absolute_error(truth_voltages, forecast_voltages)
    rmse = np.sqrt(mse)
    r2 = r2_score(truth_voltages, forecast_voltages)

    # Store metrics
    mse_list.append(mse)
    mae_list.append(mae)
    rmse_list.append(rmse)
    r2_list.append(r2)

    # Print per-file metrics
    print(f" MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(truth_voltages, label='Actual', color='red')
    plt.plot(forecast_voltages, label='Forecast', color='blue')
    plt.title(f'Forecast vs Actual: {filename}')
    plt.xlabel('Time Steps')
    plt.ylabel('Voltage')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Final Averages ===
print("\n Average Performance Across All Files:")
print(f"Avg MSE:  {np.mean(mse_list):.6f}")
print(f"Avg MAE:  {np.mean(mae_list):.6f}")
print(f"Avg RMSE: {np.mean(rmse_list):.6f}")
print(f"Avg R²:   {np.mean(r2_list):.6f}")


