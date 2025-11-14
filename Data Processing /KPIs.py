import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r'C:\Users\47406\Downloads\Coding 2023\Article- Neural networks\Neural networks- literature\feature_selection')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
import embedded_method

dataset_train = "Insert your training path"
dataset_valid="Insert your validation path"

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
