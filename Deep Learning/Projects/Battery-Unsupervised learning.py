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

data_path = 'INCLUDE HERE YOUR PATH' #the file is a dictionary saved in npy format

isExist = os.path.exists(data_path)
print(isExist)
Battery= pd.read_csv(data_path)
time= Battery.loc[:, ['start_time']]
df= Battery
import pandas as pd
from datetime import datetime
df['start_time']=pd.to_datetime(df['start_time'], format='ISO8601')
dff = df.sort_values('start_time')

#Considering only cycles and capacity for the RUL

discharge= dff.loc[dff['type'] == 'discharge']
charge= dff.loc[dff['type'] == 'charge']
d_cap= discharge.loc[:, ['start_time', 'Capacity']]
cycles= (d_cap['Capacity'].ne(d_cap['Capacity'].shift())).cumsum()
plt.axhline(y=0.7*2, color='r', linestyle='-')
plt.plot(cycles, d_cap['Capacity'])
plt.xlabel('Cycles')
plt.ylabel('Capacity')
plt.show()

#%% EDA
f_charge= charge.drop(columns =['start_time', 'type', 'ambient_temp', 'Capacity'])
f_discharge= discharge.drop(columns =['start_time', 'type', 'ambient_temp'])
f_discharge['cycles']= cycles
import seaborn as sns
sns.heatmap(f_discharge.corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);
print(f_discharge.isnull().sum())

#%% LOF, USING LOF SCORE, threshold

from sklearn.model_selection import train_test_split
from pyod.models.lof import LOF
x= np.array(f_discharge.loc[:, f_discharge.columns != 'Capacity'])
y= f_discharge.loc[:,'Capacity']
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
plt.xlabel('cycles')
plt.ylabel('capacity')
plt.scatter(np.random.choice(np.array(cycles)[ind_lof], size=500, replace=False),
            np.random.choice(np.array(y)[ind_lof], size=500, replace=False))

