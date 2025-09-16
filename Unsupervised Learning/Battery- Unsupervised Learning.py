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

#%%MULTIPLE MODELS FOR LOF
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
plt.xlabel('cycles')
plt.ylabel('capacity')
plt.scatter(np.random.choice(np.array(cycles), size=500, replace=False),
            np.random.choice(np.array(y)[f_ind_lof], size=500, replace=False))

#%% LOF SECOND APPROACH OUTLIERS

n_neighbors = [5, 20]
contamination_arr = [0.01,0.03]
#flags_arr = [1,2]
f_discharge['cycles']=cycles

from sklearn.neighbors import LocalOutlierFactor
#for f in flags_arr:
for n in n_neighbors:
    for c in contamination_arr:
            # model specification
        model1 = LocalOutlierFactor(n_neighbors = n, metric = "manhattan", contamination = c)
            # model fitting
        y_pred = model1.fit_predict(f_discharge)
            # filter outlier index
        outlier_index = np.where(y_pred == -1) # negative values are outliers and positives inliers
            # filter outlier values
        outlier_values = f_discharge.iloc[outlier_index]
            # plot data
        #plt.scatter(f_discharge["cycles"].sample(n=500), f_discharge["Capacity"].sample(n=500), color = "b", s = 65)
            # plot outlier values
        plt.scatter(outlier_values["cycles"].sample(n=500), outlier_values["Capacity"].sample(n=500), color = "r")
        plt.xlabel('cycles')
        plt.ylabel('capacity')
        plt.title('with n_neighbors = '+ str(n)+' having contamination of '+str(c))
        plt.show()

#PARAMETER OPTIMIZATION DBSCAN, INITIALIZATION

st = StandardScaler()
stdDf =  pd.DataFrame(st.fit_transform(f_discharge), columns=f_discharge.columns)
nn = NearestNeighbors(n_neighbors=5).fit(stdDf)
distances, indices = nn.kneighbors(stdDf)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
#plt.figure(figsize=(10,8))
plt.title('Eps determination')
plt.xlabel('Points sample')
plt.ylabel('5-Nearest Neighbor distance')
plt.axhline(y=0.12, color='r', linestyle='-')
plt.plot(distances)

#%% OPTICS, IMPLEMENTATION BASED ON OPTIMIZED PARAMETERS
from sklearn.cluster import OPTICS
from numpy import quantile, where, random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


model = OPTICS(eps = 0.12, min_samples = 22).fit(stdDf)
print(model)
scores = model.core_distances_
thresh = quantile(scores, .98)
print(thresh)
index = where(scores >= thresh)

np.array(y)[index]
plt.scatter(np.random.choice(np.array(cycles)[index], size=500, replace=False),
            np.random.choice(np.array(y)[index], size=500, replace=False), color= 'b')
plt.title('OPTICS outliers')
plt.xlabel('cycles')
plt.ylabel('capacity')

#DBSCAN IMPLEMENTATION, BASED ON PREVIOUS PARAMETERS
dbsc = DBSCAN(eps = 0.12, min_samples = 22).fit(stdDf)
labels = dbsc.labels_
out_df = f_discharge.copy()
out_df['label'] = dbsc.labels_
out_df['label'].value_counts()

out_df['cycles']= cycles
dbsc_out= out_df[out_df['label']==-1]
plt.scatter(dbsc_out['cycles'].sample(n=500), (dbsc_out['Capacity'].sample(frac=1)).sample(n=500), color= 'green')
plt.title('DBSCAN outliers')
plt.xlabel('cycles')
plt.ylabel('capacity')


fig, (ax1, ax2, ax3) = plt.subplots(3)
plt.subplots_adjust(left=0.15,
                    bottom=0.15,
                    right=0.9,
                    top=0.85,
                    wspace=1,
                    hspace=1)
fig.suptitle('Outlier analysis')
ax1.scatter(outlier_values["cycles"].sample(n=200), outlier_values["Capacity"].sample(n=200), color = "r")
ax1.set_ylim(1,2)
#ax1.set_xlim(0.1,1.1)
ax2.scatter(dbsc_out['cycles'].sample(n=200), (dbsc_out['Capacity'].sample(frac=1)).sample(n=200), color= 'b')
ax2.set_ylim(1,2)
#ax2.set_xlim(0.1,1.1)
ax3.scatter(np.random.choice(np.array(cycles)[index], size=200, replace=False),
            np.random.choice(np.array(y)[index], size=200, replace=False))
ax3.set_ylim(1,2)
#ax3.set_xlim(0.1,1.1)
ax1.set_title('LOF')
ax2.set_title('DBSCAN')
ax3.set_title('OPTICS')
fig.supxlabel('Cycles')
fig.supylabel('Capacity')
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
