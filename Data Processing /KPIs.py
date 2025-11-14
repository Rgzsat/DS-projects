import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r'C:\Users\Downloads\Coding 2023\Article- Neural networks\Neural networks- literature\feature_selection')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
import embedded_method

dataset_train = "Insert your training path"
dataset_valid="Insert your validation path"

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

def df(dataset):
    dataset=(dataset.drop('wh', axis=1))
    dataset = dataset.rename(columns={'mAh': 'Capacity', 'V': 'Voltage', 
                                      'R': 'Resistance'})
    cap= dataset['Capacity']/1000
    max_cap=cap[len(cap)-1]
    dod= (max_cap - cap)/max_cap
    dataset['SOC']= dod
    X= dataset
    
    ir= (4.1-dataset['Voltage'])/dataset['I']
    yi= ir[0]*dataset['I']+dataset['Voltage']
    
    return X, yi
