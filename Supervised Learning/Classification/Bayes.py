import numpy as np
import math

from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import roc_auc_score

#data = datasets.load_breast_cancer()
#X, y = data.data, data.target
X, y = datasets.make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

n_samples, n_features = X.shape
classes = np.unique(y)
n_classes = len(classes)
ini_mean = np.zeros((n_classes, n_features), dtype=np.float64)
ini_var = np.zeros((n_classes, n_features), dtype=np.float64)
prior_prob =  np.zeros(n_classes, dtype=np.float64)
#%%

def initial_calc(X,y, classes):
    
#Calculation of the mean and variance for each class  
    for i, c in enumerate(classes):
        X_for_class_c = X[y==c]
        ini_mean[i, :] = X_for_class_c.mean(axis=0)
        ini_var[i, :] = X_for_class_c.var(axis=0)
        prior_prob[i] = X_for_class_c.shape[0] / float(n_samples)
    
    return np.concatenate((ini_mean, ini_var), axis=0), prior_prob

f_pos= initial_calc(X, y, classes)

