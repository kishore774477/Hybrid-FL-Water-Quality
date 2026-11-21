"""
Comparison Script for Baseline Models vs Proposed Hybrid FL Model.
Trains Baseline Models (Logistic Regression, Random Forest) on the exact
union of client datasets to ensure fairness, then compares against the
global model produced by the FL approach.
"""

import numpy as np
import pandas as pd
import random
import torch
import sys
import os
from sklearn.model_selection import train_test_split

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models_baseline.logistic_regression import run_logistic_regression
from models_baseline.random_forest import run_random_forest
from models_baseline.proposed_model_eval import evaluate_proposed_model
from model.base_model import WaterQualityModel
from explainability.feature_importance import permutation_importance
from config import CONFIG

# -----------------------------
# Utility
# -----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# -----------------------------
# Load dataset
# -----------------------------
TARGET_COL = CONFIG["TARGET_COL"]
SEEDS = CONFIG["SEEDS"]

def get_path(rel_path):
    return os.path.join(os.path.dirname(__file__), rel_path)

# 1. Load Global Test Set (Fixed once)
test_path = get_path("../data/processed/test_data_global.csv")
df_test = pd.read_csv(test_path)
X_test_global = df_test.drop(columns=[TARGET_COL]).values
y_test_global = df_test[TARGET_COL].values

# 2. Load Training Splits (Train Pool - Union of all clients)
client_files = [
    "../data/splits/client_industrial_1.csv",
    "../data/splits/client_industrial_2.csv",
    "../data/splits/client_residential_1.csv",
    "../data/splits/client_residential_2.csv",
    "../data/splits/client_agricultural_1.csv",
    "../data/splits/client_agricultural_2.csv"
]
df_train_list = [pd.read_csv(get_path(f)) for f in client_files]
df_train_full = pd.concat(df_train_list, ignore_index=True)

X_train_full = df_train_full.drop(columns=[TARGET_COL]).values
y_train_full = df_train_full[TARGET_COL].values

# Extract feature names for explainability
feature_names = df_train_full.drop(columns=[TARGET_COL]).columns.tolist()

# -----------------------------
# Storage
# -----------------------------
results = {
    "LogisticRegression": [],
    "RandomForest": [],
    "ProposedHybridFL": []
}

# -----------------------------
# Experiments Loop
# -----------------------------
for seed in SEEDS:
    print(f"\nRunning seed {seed}")
    set_seed(seed)

    # Further split training data just to add randomness to baselines, 
    # but still on the FAIR client dataset union
    X_train, _, y_train, _ = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=seed
    )

    # Logistic Regression
    # Save model only on the last seed to avoid repeated I/O
    save_lr = get_path("../saved_models/logistic_regression.pkl") if seed == SEEDS[-1] else None
    acc_lr = run_logistic_regression(
        X_train, y_train, X_test_global, y_test_global, save_path=save_lr
    )
    results["LogisticRegression"].append(acc_lr)

    # Random Forest
    save_rf = get_path("../saved_models/random_forest.pkl") if seed == SEEDS[-1] else None
    acc_rf = run_random_forest(
        X_train, y_train, X_test_global, y_test_global, seed, save_path=save_rf
    )
    results["RandomForest"].append(acc_rf)

    # -----------------------------
    # Proposed Model (load trained model)
    # -----------------------------
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(np.concatenate([y_train_full, y_test_global])))
    proposed_model = WaterQualityModel(input_dim, output_dim=num_classes)
    
    model_path = get_path("../global_model.pth")
    if os.path.exists(model_path):
        proposed_model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )
    else:
        print("Warning: global_model.pth not found. Using random weights.")

    acc_prop = evaluate_proposed_model(
        proposed_model, X_test_global, y_test_global
    )
    results["ProposedHybridFL"].append(acc_prop)

# -----------------------------
# Feature Importance Analysis
# -----------------------------
print("\nRunning Feature Importance Analysis...")

importances = permutation_importance(
    proposed_model,
    X_test_global,
    y_test_global,
    feature_names
)

print("\nFeature Importance (Permutation-Based):")
for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{k}: {v:.4f}")

# -----------------------------
# Final results
# -----------------------------
print("\n===== FINAL RESULTS (Mean ± Std) =====")
for model_name, scores in results.items():
    mean = np.mean(scores) * 100
    std = np.std(scores) * 100
    print(f"{model_name}: {mean:.2f}% ± {std:.2f}%")
