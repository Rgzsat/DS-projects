
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

import seaborn as sns

df= Battery
df['time']= time_exp
sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);
print(df.isnull().sum())


from sklearn.model_selection import train_test_split
from pyod.models.lof import LOF
x= np.array(df.loc[:, df.columns != 'V'])
y= df.loc[:,'mAh']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

from pyod.models.lof import LOF
lof = LOF(contamination=0.10)
lof.fit(X_train)

# Training data
y_train_scores = lof.decision_function(X_train)
y_train_pred = lof.predict(X_train)

# Test data
y_test_scores = lof.decision_function(X_test)
y_test_pred = lof.predict(X_test) # outlier labels (0 or 1)

def count_stat(vector):
    # Because it is '0' and '1', we can run a count statistic.
    unique, counts = np.unique(vector, return_counts=True)
    return dict(zip(unique, counts))

print("The training data:", count_stat(y_train_pred))
print("The testing data:", count_stat(y_test_pred))
# Threshold for the defined comtanimation rate
print("The threshold for the defined comtanimation rate:" , lof.threshold_)
lof.get_params()

import matplotlib.pyplot as plt
plt.hist(y_train_scores, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram LOF")
plt.xlabel('LOF outlier score')
plt.ylabel('Frequency')
plt.xlim(0.95,1.05)
plt.show()
threshold = lof.threshold_ # Or other value from the above histogram
print((np.array(np.where(y_test_scores>=threshold))).shape)

ind_lof= np.where(y_test_scores>=threshold)
np.array(y)[np.where(y_test_scores>=threshold)]
np.array(y)[ind_lof]
plt.title('LOF intial outliers')
plt.xlabel('Time')
plt.ylabel('capacity')
plt.scatter(np.random.choice(np.array(time_exp)[ind_lof], size=30, replace=False),
            np.random.choice(np.array(y)[ind_lof], size=30, replace=False))
