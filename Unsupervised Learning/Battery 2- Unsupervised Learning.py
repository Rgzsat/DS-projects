
import numpy as np
import random
import os
import scipy.io
import matplotlib.pyplot as plt
from math import sqrt
from datetime import datetime
import pandas as pd
from pathlib import Path
import os
import re

data_path = #LOAD YOUR PATH HERE

# 🔹 Function to Convert HH:MM:SS to Seconds
def clean_time_format(time_str):
    time_str = re.sub(r'\s+', '', str(time_str))  # Remove spaces
    time_parts = time_str.split(":")
    if len(time_parts) == 3:
        h, m, s = map(int, time_parts)
        return h * 3600 + m * 60 + s
    elif len(time_parts) == 2:
        m, s = map(int, time_parts)
        return m * 60 + s
    else:
        raise ValueError(f"Invalid time format: {time_str}")

isExist = os.path.exists(data_path)
print(isExist)

Battery= pd.read_csv(data_path)
#data = pd.read_csv(file)  # Assuming "time" is the index
time_exp = np.array([clean_time_format(t) for t in Battery["time"]])  # Convert time to seconds
voltage_exp = np.array(Battery["V"])
import pandas as pd
from datetime import datetime

plt.plot(time_exp, voltage_exp)
plt.xlabel('Time [sec]')
plt.ylabel('Voltage [V]')
plt.show()
