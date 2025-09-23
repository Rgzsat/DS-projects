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
