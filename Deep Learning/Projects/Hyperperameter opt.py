
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Dropout, Flatten
from sklearn.preprocessing import MinMaxScaler
# Machine learning algorithms
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import train_test_split

dataset_train = "INSERT YOUR TRAININT PATH HERE"

dataset_valid= "INSERT YOUR VALIDATION PATH HERE"

dataset_test="INSERT YOUR TESTING PATH HERE"

#OPTIONAL, IF YOU HAVE LIMITED DATASETS, IT CONTAINS ALSO A FUNCTION TO SPLIT DATA

train_size = int(len(dataset_train) * 0.70)
val_size= int(len(dataset_train) * 0.20)
test_size= int(len(dataset_train) * 0.10)
scaler = StandardScaler()

def get_val(dataset):
    dataset=(dataset.drop('time', axis=1))
    cap= dataset['mAh']/1000
    max_cap=cap[len(cap)-1]
    dod= (max_cap - cap)/max_cap
    dataset['SOC']= dod
    X= dataset
    X= scaler.fit_transform(X)
    
    ir= (4.1-dataset['V'])/dataset['I']
    yi= ir[0]*dataset['I']+dataset['V']
    y = np.array(yi)
    
    return X,y

X_train, y_train=get_val(dataset_train)[0], get_val(dataset_train)[1]
X_val, y_val=get_val(dataset_valid)[0], get_val(dataset_valid)[1]
X_test, y_test=get_val(dataset_test)[0], get_val(dataset_test)[1]

X_train, y_train=get_val(dataset_train)[0], get_val(dataset_train)[1]
X_val, y_val=get_val(dataset_valid)[0], get_val(dataset_valid)[1]
X_test, y_test=get_val(dataset_test)[0], get_val(dataset_test)[1]


#X_train = scaler.fit_transform(X_train)
#X_val = scaler.fit_transform(X_val)
#X_test = scaler.fit_transform(X_test)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
