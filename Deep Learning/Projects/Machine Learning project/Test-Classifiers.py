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

# ---------------------- Evaluate Model ----------------------

def evaluate_with_threshold(y_true, y_probs, threshold=0.5):
    y_pred_thresh = (y_probs >= threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_thresh, target_names=["Good", "Bad"]))

    accuracy = accuracy_score(y_true, y_pred_thresh)
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else None
    f2 = fbeta_score(y_true, y_pred_thresh, beta=2)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc:.4f}" if auc else "ROC AUC: N/A")
    print(f"F2 Score: {f2:.4f}")

    cm = confusion_matrix(y_true, y_pred_thresh, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    return accuracy, auc, f2

# ---------------------- Precision-Recall Curve ----------------------

def plot_precision_recall(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-9)
    best_idx = np.argmax(f2_scores)
    best_thresh = thresholds[best_idx]

    plt.plot(recall, precision, label='PR Curve')
    plt.scatter(recall[best_idx], precision[best_idx], color='red', label=f'Best F2 @ {best_thresh:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f" Best threshold by F2 score: {best_thresh:.2f}")
    return best_thresh

# ---------------------- Cross-Validation ----------------------

def run_cross_validation(X, y, model=None, n_splits=5, test_size=0.2):
    if model is None:
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    accuracies, roc_aucs = [], []

    print(f"\n Class distribution: {dict(pd.Series(y).value_counts())}\n")

    for fold, (train_idx, test_idx) in enumerate(sss.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f" Fold {fold} class distribution train: {y_train.value_counts().to_dict()} test: {y_test.value_counts().to_dict()}")

        if len(y_test.unique()) < 2:
            print("⚠️ Only one class in test set. Skipping fold.")
            continue

        min_class = y_train.value_counts().idxmin()
        min_count = y_train.value_counts().min()
        safe_k = max(1, min(5, min_count - 1))

        if min_count > 1:
            try:
                smote = SMOTE(random_state=42, k_neighbors=safe_k)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            except ValueError as e:
                warnings.warn(f"SMOTE failed: {e}. Using original training set.")
                X_train_resampled, y_train_resampled = X_train.copy(), y_train.copy()
        else:
            print(" Not enough samples for SMOTE. Using original training set.")
            X_train_resampled, y_train_resampled = X_train.copy(), y_train.copy()

        model.fit(X_train_resampled, y_train_resampled)

        try:
            y_probs = model.predict_proba(X_test)[:, 1]
        except:
            y_probs = model.predict(X_test)

        threshold = plot_precision_recall(y_test, y_probs)
        acc, auc, f2 = evaluate_with_threshold(y_test, y_probs, threshold)

        accuracies.append(acc)
        if auc is not None:
            roc_aucs.append(auc)

    print(f"\n Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    if roc_aucs:
        print(f" Average ROC AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    else:
        print(" ROC AUC could not be calculated in any fold.")

# ---------------------- Main ----------------------

if __name__ == "__main__":
    path = "battery_discharge_features-INR21700.csv"
    X, y = load_data(path, capacity_threshold=capacity_threshold)
    X = select_top_features(X, y, k=10)

    if y.value_counts().min() < 5:
        one_class_loocv(X, y)
    else:
        run_cross_validation(X, y)
