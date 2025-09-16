
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


from pyod.models.combination import aom, moa, average, maximization
from pyod.utils.utility import standardizer
from pyod.models.lof import LOF
# Standardize data
X_train_norm, X_test_norm = standardizer(X_train, X_test)
# Test a range of k-neighbors from 10 to 200. There will be 20 models.
n_clf = 20
k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
 120, 130, 140, 150, 160, 170, 180, 190, 200]
# Just prepare data frames so we can store the model results
train_scores = np.zeros([X_train.shape[0], n_clf])
test_scores = np.zeros([X_test.shape[0], n_clf])
train_scores.shape
# Modeling
for i in range(n_clf):
    k = k_list[i]
    lof = LOF(n_neighbors=k)
    lof.fit(X_train_norm)

    # Store the results in each column:
    train_scores[:, i] = lof.decision_scores_
    test_scores[:, i] = lof.decision_function(X_test_norm)
# Decision scores have to be normalized before combination
train_scores_norm, test_scores_norm = standardizer(train_scores,test_scores)

# Combination by average
# The test_scores_norm is 500 x 20. The "average" function will take the average of the 20 columns. The result "y_by_average" is a single column:
y_by_average = average(train_scores_norm)
import matplotlib.pyplot as plt
plt.hist(y_by_average, bins='auto') # arguments are passed to np.histogram
plt.title("Combination by average")
plt.xlabel('LOF score')
plt.ylabel('Frequency')
plt.xlim(-0.7,1.5)
plt.show()

mult_lof= test_scores.mean(axis=1)
f_ind_lof= np.where(mult_lof>=threshold)

plt.title('LOF models outliers')
plt.xlabel('time')
plt.ylabel('capacity')
plt.scatter(np.random.choice(np.array(time_exp), size=30, replace=False),
            np.random.choice(np.array(y)[f_ind_lof], size=30, replace=False))

##SECOND APPROACH, USING LOF NEIGHBORS AND CONTAMINATION 

from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

n_neighbors = [5, 20]
contamination_arr = [0.01, 0.03]

for n in n_neighbors:
    for c in contamination_arr:
        # model specification
        model1 = LocalOutlierFactor(n_neighbors=n, metric="manhattan", contamination=c)

        # model fitting
        y_pred = model1.fit_predict(df.loc[:, df.columns != 'V'])

        # filter outlier index (where y_pred == -1 means outliers)
        outlier_index = np.where(y_pred == -1)

        # filter outlier values
        outlier_values = df.loc[:, df.columns != 'V'].iloc[outlier_index]

        # Check if we have enough samples in outlier_values
        if len(outlier_values) >= 500:
            # Plot data (outliers)
            plt.scatter(outlier_values["time"].sample(n=500), outlier_values["mAh"].sample(n=500), color="r")
        else:
            # If not enough, sample the available outliers
            plt.scatter(outlier_values["time"], outlier_values["mAh"], color="r")

        # Set labels and title
        plt.xlabel('Time')
        plt.ylabel('mAh')
        plt.title(f'Outlier Points with n_neighbors={n} and contamination={c}')
        plt.show()

        plt.show()
