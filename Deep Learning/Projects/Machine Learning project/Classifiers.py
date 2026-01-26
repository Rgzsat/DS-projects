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

# ---------------------- Feature Engineering ----------------------

def extract_features(df):
    df['time_sec'] = df['time'].apply(parse_time_column)

    features = {
        'max_current': df['I'].max(),
        'avg_current': df['I'].mean(),
        'min_voltage': df['V'].min(),
        'max_voltage': df['V'].max(),
        'avg_voltage': df['V'].mean(),
        'total_capacity_mAh': df['mAh'].iloc[-1],
        'total_energy_Wh': df['wh'].iloc[-1],
        'total_discharge_time_sec': df['time_sec'].iloc[-1],
        'avg_power': df['P'].mean(),
        'resistance_std': df['R'].std()
    }

    # Additional features
    features.update({
        'voltage_std': df['V'].std(),
        'current_skew': skew(df['I']),
        'current_kurtosis': kurtosis(df['I']),
        'voltage_drop_rate': (df['V'].iloc[0] - df['V'].iloc[-1]) / df['time_sec'].iloc[-1],
        'capacity_per_sec': df['mAh'].iloc[-1] / df['time_sec'].iloc[-1],
        'energy_per_sec': df['wh'].iloc[-1] / df['time_sec'].iloc[-1],
        'max_power': df['P'].max()
    })

    # Rolling statistics
    df['V_roll_std'] = df['V'].rolling(window=5, min_periods=1).std()
    df['I_roll_std'] = df['I'].rolling(window=5, min_periods=1).std()
    df['P_roll_std'] = df['P'].rolling(window=5, min_periods=1).std()

    features.update({
        'V_roll_std_mean': df['V_roll_std'].mean(),
        'I_roll_std_mean': df['I_roll_std'].mean(),
        'P_roll_std_mean': df['P_roll_std'].mean()
    })

    return features

# ---------------------- Folder Processing ----------------------

def process_folder(folder_path):
    all_features = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath)

            if df.empty or df.isnull().any().any():
                print(f" Skipping file with issues: {filename}")
                continue

            features = extract_features(df)
            features['file'] = filename
            all_features.append(features)

    return pd.DataFrame(all_features)


# ---------------------- Visualization ----------------------

def visualize_data(features_df):
    plt.figure(figsize=(10, 6))
    sns.histplot(features_df['avg_current'], kde=True)
    plt.title('Distribution of Average Current')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=features_df, x='avg_voltage', y='total_capacity_mAh', hue='total_discharge_time_sec')
    plt.title('Voltage vs Capacity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    numeric_df = features_df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# ---------------------- Feature Selection ----------------------

def prepare_data_for_selection(features_df, target_col='label'):
    features_df = features_df.dropna()
    X = features_df.select_dtypes(include=[float, int]).drop(columns=[target_col])
    y = features_df[target_col]
    return X, y

def filter_method(X, y, k=10):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X, y)
    selected = X.columns[selector.get_support()]
    scores = selector.scores_

    print("\n Filter Method (Top K by Mutual Info):")
    for name, score in zip(X.columns, scores):
        print(f"{name}: {score:.4f}")
    return selected

def wrapper_method(X, y, estimator=None, n_features=10):
    if estimator is None:
        estimator = LogisticRegression(max_iter=1000)
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    selected = X.columns[rfe.support_]

    print("\n Wrapper Method (RFE):")
    for name, rank in zip(X.columns, rfe.ranking_):
        print(f"{name}: Rank {rank}")
    return selected

def embedded_method(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    selected = X.columns[sorted_idx[:10]]

    print("\n Embedded Method (Random Forest):")
    for name, score in zip(X.columns[sorted_idx], importances[sorted_idx]):
        print(f"{name}: {score:.4f}")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[sorted_idx[:10]], y=X.columns[sorted_idx[:10]])
    plt.title("Top 10 Features - Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    return selected

# ---------------------- Main ----------------------

if __name__ == "__main__":
    folder = "/content/drive/MyDrive/Writing workshop data/Final data"
    features_df = process_folder(folder)

    print(f"\n Processed {len(features_df)} files.\n")
    print(features_df.head())

    # Simulated binary label (you should replace this with your own target)
    features_df['label'] = (features_df['total_capacity_mAh'] > 0.8*4900).astype(int)

    features_df.to_csv("battery_discharge_features-INR21700.csv", index=False)
    print("\n Features saved to battery_discharge_features-INR21700.csv")

    # Visualize the data
    visualize_data(features_df)

    # Feature Selection
    X, y = prepare_data_for_selection(features_df, target_col='label')

    selected_filter = filter_method(X, y, k=10)
    selected_wrapper = wrapper_method(X, y, n_features=10)
    selected_embedded = embedded_method(X, y)

# ======================================================
#  Battery Pack Diagnostics: Voltage–SOC + R_int Method
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import glob
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
