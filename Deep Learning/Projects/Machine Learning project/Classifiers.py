import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ---------------------- Time Parser ----------------------

def parse_time_column(time_str):
    try:
        h, m, s = map(int, time_str.strip().split(":"))
        return h * 3600 + m * 60 + s
    except Exception as e:
        print(f"Time parsing error: {time_str} -> {e}")
        return np.nan
