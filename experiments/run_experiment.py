import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import copy
import numpy as np
import random
import json

# Ensure imports work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)

from model.base_model import WaterQualityModel
from evaluation.metrics import evaluate_model
from clients.local_training import local_train
from aggregation.fedavg import fedavg
from clients.uncertainty import compute_uncertainty
from aggregation.uncertainty_weighted import uncertainty_weighted_aggregation
from aggregation.hierarchical_agg import hierarchical_aggregation
from hybrid.hybrid_controller import hybrid_control
from config import CONFIG

# --- FORMAL EXPERIMENTAL PROTOCOL ---
SEEDS = CONFIG["SEEDS"]
ROUNDS = CONFIG["ROUNDS"]
LOCAL_EPOCHS = CONFIG["LOCAL_EPOCHS"]
TARGET_COL = CONFIG["TARGET_COL"]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_path(path):
    if os.path.exists(path): return path
    alt = os.path.join(os.path.dirname(__file__), "..", path)
    if os.path.exists(alt): return alt
    raise FileNotFoundError(f"Could not find {path}")

# Load GLOBAL TEST SET (Held-out)
test_path = get_path("data/processed/test_data_global.csv")
df_test = pd.read_csv(test_path)
X_test = torch.tensor(df_test.drop(columns=[TARGET_COL]).values, dtype=torch.float32)
y_test = torch.tensor(df_test[TARGET_COL].values, dtype=torch.long)
num_classes = len(set(df_test[TARGET_COL].values))

# Load TRAINING SPLITS (Train Pool)
clients_data = {
    "industrial_1": pd.read_csv(get_path("data/splits/client_industrial_1.csv")),
    "industrial_2": pd.read_csv(get_path("data/splits/client_industrial_2.csv")),
    "residential_1": pd.read_csv(get_path("data/splits/client_residential_1.csv")),
    "residential_2": pd.read_csv(get_path("data/splits/client_residential_2.csv")),
    "agricultural_1": pd.read_csv(get_path("data/splits/client_agricultural_1.csv")),
    "agricultural_2": pd.read_csv(get_path("data/splits/client_agricultural_2.csv"))
}

# Reference for Drift: the pooled training distribution across all clients.
# Each region's current data is compared against it during periodic checks.
reference_df = pd.concat(clients_data.values(), ignore_index=True)

def run_trial(seed):
    print(f"Running Seed {seed}...", flush=True)
    set_seed(seed)
    
    # Initialize Global Model
    input_dim = X_test.shape[1]
    global_model = WaterQualityModel(input_dim, output_dim=num_classes)
    
    regions = {
        "industrial": ["industrial_1", "industrial_2"],
        "residential": ["residential_1", "residential_2"],
        "agricultural": ["agricultural_1", "agricultural_2"]
    }

    final_acc = 0.0
    
    # Logging objects
    log_accuracies = []
    log_drift_events = []
    log_drift_checks = []
    log_weights = []

    for r in range(ROUNDS):
        region_models = {}
        round_weights = {}

        for region, client_names in regions.items():
            state_dicts = []
            uncertainties = []

            for cname in client_names:
                df_client = clients_data[cname]

                # Local Training
                local_state = local_train(
                    global_model,
                    df_client,
                    TARGET_COL,
                    epochs=LOCAL_EPOCHS
                )
                state_dicts.append(local_state)

                # Uncertainty Estimation
                temp_model = WaterQualityModel(input_dim, output_dim=num_classes)
                temp_model.load_state_dict(local_state)
                X_client = torch.tensor(df_client.drop(columns=[TARGET_COL]).values, dtype=torch.float32)
                uncertainty = compute_uncertainty(temp_model, X_client)
                uncertainties.append(uncertainty)

            region_models[region] = (state_dicts, uncertainties)
            round_weights[region] = uncertainties

        # Hierarchical Aggregation
        global_state = hierarchical_aggregation(region_models)
        global_model.load_state_dict(global_state)
        
        # Hybrid Control (Drift Check)
        # Every 5 rounds, compare each region's data with the pooled reference.
        # A drifted region triggers centralized fine-tuning on that region's data.
        if (r+1) % 5 == 0:
            for region, client_names in regions.items():
                region_df = pd.concat([clients_data[c] for c in client_names], ignore_index=True)
                global_model, drift_flag, psi = hybrid_control(
                    global_model, reference_df, region_df, TARGET_COL, region=region
                )
                log_drift_checks.append({
                    "round": r+1, "region": region, "psi": psi, "drift": drift_flag
                })
                if drift_flag and (r+1) not in log_drift_events:
                    log_drift_events.append(r+1)

        # Evaluate model after round
        metrics = evaluate_model(global_model, X_test, y_test)
        log_accuracies.append(metrics['accuracy'])
        log_weights.append(round_weights)
        
    final_acc = log_accuracies[-1]
    
    # Save log for this seed
    log_data = {
        "seed": seed,
        "round_accuracies": log_accuracies,
        "drift_events": log_drift_events,
        "drift_checks": log_drift_checks,
        "uncertainties": log_weights
    }
    
    log_dir = os.path.join(ROOT, "experiments", "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"seed_{seed}_log.json"), "w") as f:
        json.dump(log_data, f, indent=4)
    
    print(f" Done. Acc: {final_acc:.4f}")
    return final_acc, global_model

# --- MAIN LOOP ---
results = []
best_model = None
best_acc = 0.0

print(f"Starting Formal Experiment (Seeds: {SEEDS})")
print(f"Protocol: Hybrid FL ({ROUNDS} Rounds) -> Eval on Global Test Set")

for seed in SEEDS:
    acc, model = run_trial(seed)
    results.append(acc)
    
    if acc > best_acc:
        best_acc = acc
        best_model = model

# --- STATISTICAL ANALYSIS ---
mean_acc = np.mean(results)
std_acc = np.std(results)
mean_acc_percent = mean_acc * 100
std_acc_percent = std_acc * 100

print("\n=== FINAL RESULTS ===")
print(f"Accuracy = {mean_acc_percent:.2f}% ± {std_acc_percent:.2f}%")

# Save Best Model for GUI
if best_model:
    torch.save(best_model.state_dict(), os.path.join(ROOT, "global_model.pth"))
    print("Best model saved to 'global_model.pth'")
