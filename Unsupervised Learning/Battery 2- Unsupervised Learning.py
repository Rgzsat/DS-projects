
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

#SINGLE PLOT

#single plot for LOF
modellof = LocalOutlierFactor(n_neighbors = 5, metric = "manhattan", contamination = 0.03)
    # model fitting
y_pred = modellof.fit_predict(df.loc[:, df.columns != 'V'])
    # filter outlier index
outlier_index = np.where(y_pred == -1) # negative values are outliers and positives inliers
    # filter outlier values
outlier_values = df.loc[:, df.columns != 'V'].iloc[outlier_index]
    # plot data
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

#$DBSCAN IMPLEMENTATION

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st = StandardScaler()
stdDf = pd.DataFrame(st.fit_transform(df.loc[:, df.columns != 'V']), columns=df.loc[:, df.columns != 'V'].columns)

# Compute distances to 4th nearest neighbor (5 - 1)
nn = NearestNeighbors(n_neighbors=5)
nn_fit = nn.fit(stdDf)
distances, indices = nn_fit.kneighbors(stdDf)

# Sort the distances to plot the "elbow"
distances = np.sort(distances[:, 4])  # 4 = n_neighbors - 1

plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.title("K-distance Graph (for DBSCAN eps selection)")
plt.xlabel("Points sorted by distance")
plt.ylabel("4th Nearest Neighbor Distance")
plt.grid(True)
plt.show()

from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

best_score = -1
best_params_dbscan = {}

# Try ranges of eps and min_samples
#eps_values = [0.08, 0.1, 0.12, 0.15, 0.18]
eps_values = [0.2, 0.25, 0.30]
min_samples_values = [5, 10, 15, 22, 30]

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(stdDf)

        # Ignore if only one cluster (or all noise)
        if len(set(labels)) <= 1 or len(set(labels)) == len(stdDf):
            continue

        mask = labels != -1
        if len(set(labels[mask])) <= 1:
            continue

        score = silhouette_score(stdDf[mask], labels[mask])
        print(f"eps={eps}, min_samples={min_samples}, Silhouette={score:.3f}")

        if score > best_score:
            best_score = score
            best_params_dbscan = {'eps': eps, 'min_samples': min_samples}

print("\nBest DBSCAN params:", best_params_dbscan)
print("Best Silhouette Score:", round(best_score, 3))


##DBSCAN IMPLEMENTATION, BASED ON PREVIOUS PARAMETERS
from sklearn.cluster import DBSCAN

# Replace with your best values
dbscan = DBSCAN(eps=best_params_dbscan['eps'], min_samples=best_params_dbscan['min_samples'])
labels = dbscan.fit_predict(stdDf)

# Plot
dbsc_out_df = df.loc[:, df.columns != 'V'].copy()
dbsc_out_df['label'] = labels
dbsc_out_df['time'] = time_exp
outliers_db = dbsc_out_df[dbsc_out_df['label'] == -1]

# Sample size
sample_size = min(200, len(outliers_db))

plt.scatter(
    outliers_db['time'].sample(n=sample_size),
    outliers_db['mAh'].sample(n=sample_size),
    color='purple'
)
plt.title('Optimized DBSCAN Outliers')
plt.xlabel('Time')
plt.ylabel('mAh')
plt.grid(True)
plt.show()

##OPTICS IMPLEMENTATION

from sklearn.cluster import OPTICS
from numpy import quantile, where, random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --- Standardize Data ---
st = StandardScaler()
stdDf = pd.DataFrame(st.fit_transform(df.loc[:, df.columns != 'V']),
                     columns=df.loc[:, df.columns != 'V'].columns)

# --- Parameter Grid for OPTICS ---
min_samples_range = range(5, 70, 5)  # Try min_samples = 5, 10, ..., 45
eps_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.40]  # Try various eps values
best_score = -1
best_params_optics = {}

# --- Optimization Loop ---
for min_s in min_samples_range:
    for eps in eps_range:
        model = OPTICS(eps=eps, min_samples=min_s)
        labels = model.fit_predict(stdDf)

        # Mask noise points
        mask = labels != -1
        n_clusters = len(np.unique(labels[mask]))

        if n_clusters > 1:
            score = silhouette_score(stdDf[mask], labels[mask])
            print(f"eps={eps}, min_samples={min_s} → silhouette score: {score:.3f}")

            if score > best_score:
                best_score = score
                best_params_optics = {'eps': eps, 'min_samples': min_s}
        else:
            print(f"eps={eps}, min_samples={min_s} → not enough clusters")

print(f"\n Best OPTICS params: {best_params_optics} with silhouette score: {best_score:.3f}")

# --- Final OPTICS Model with Best Params ---
optics_model = OPTICS(eps=best_params_optics['eps'], min_samples=best_params_optics['min_samples'])
labels_optics = optics_model.fit_predict(stdDf)

# Get core distances
scores = optics_model.core_distances_

# Determine threshold by 98th percentile of core distances
thresh = np.quantile(scores, 0.98)
print(f"Outlier threshold (98th percentile): {thresh:.4f}")

# Get indices of outliers (core distances above threshold)
index = np.where(scores >= thresh)
outliers_optics = np.array(y)[index]

# Sample size control
sample_size = min(500, len(index[0]))

# Plotting
plt.scatter(
    np.random.choice(np.array(time_exp)[index], size=sample_size, replace=False),
    np.random.choice(outliers_optics, size=sample_size, replace=False),
    color='g'
)
plt.title(f'OPTICS Outliers (eps={best_params_optics["eps"]}, min_samples={best_params_optics["min_samples"]})')
plt.xlabel('Time')
plt.ylabel('Capacity (mAh)')
plt.show()
