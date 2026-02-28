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

#In this part, please select your total battery capacity and the desired threshold

#The threshold is related to an End-of-Life criteria for a battery
capacity_threshold = 0.70 * 4900  # = 3920

# ---------------------- Load and Prepare ----------------------

from sklearn.svm import OneClassSVM

def one_class_loocv(X, y):
    X_good = X[y == 0]
    X_bad = X[y == 1]

    print(f"\n One-Class SVM Evaluation: {len(X_bad)} 'Bad' cells to test")

    model = OneClassSVM(kernel="rbf", gamma='scale', nu=0.05)
    model.fit(X_good)

    preds = model.predict(X_bad)
    preds = np.where(preds == -1, 1, 0)  # -1 → anomaly → "Bad"

    y_true = np.ones(len(X_bad))  # All true labels = "Bad"

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"])
    plt.title('One-Class SVM - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, preds, target_names=["Good", "Bad"]))

def load_data(path, target_col='label', capacity_threshold=capacity_threshold):
    df = pd.read_csv(path)

    if target_col not in df.columns:
        # Label: 1 = Bad (below threshold), 0 = Good
        df[target_col] = (df['total_capacity_mAh'] < capacity_threshold).astype(int)

    print(" Class distribution:", df[target_col].value_counts().to_dict())

    df = df.dropna()
    X = df.select_dtypes(include=[float, int]).drop(columns=[target_col])
    y = df[target_col]
    return X, y

# ---------------------- Feature Selection ----------------------

def select_top_features(X, y, k=10):
    mi = mutual_info_classif(X, y)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    print("\n Top features by mutual information:")
    print(mi_series.head(k))
    return X[mi_series.head(k).index]
