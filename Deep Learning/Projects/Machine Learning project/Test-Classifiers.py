import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, fbeta_score
)
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

capacity_threshold = 0.70 * 4900  # = 3920
