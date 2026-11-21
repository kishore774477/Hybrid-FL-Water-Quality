import glob
import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.preprocessing import StandardScaler

# Ensure imports work no matter where Streamlit is launched from
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from model.base_model import WaterQualityModel
from evaluation.confidence_eval import confidence_decision
from explainability.feature_importance import permutation_importance
from drift_detection.drift_detector import detect_drift
from config import CONFIG


def root_path(*parts):
    return os.path.join(ROOT, *parts)


TARGET_COL = CONFIG["TARGET_COL"]

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Hybrid FL Water Quality Monitoring",
    layout="wide"
)

st.title("💧 Hybrid Federated Learning–Based Water Quality Monitoring")
st.markdown(
    "Uncertainty-Aware • Hierarchical • Drift-Adaptive • Confidence-Based Decision Support"
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ System Controls")

st.sidebar.subheader("🧪 Model Selection")
model_choice = st.sidebar.selectbox(
    "Select Model for Prediction",
    [
        "Proposed Hybrid FL Model",
        "Logistic Regression",
        "Random Forest"
    ]
)

mode = st.sidebar.selectbox(
    "Training Mode",
    ["Federated Learning", "Hybrid Federated Learning"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.4,
    max_value=0.9,
    value=CONFIG["CONFIDENCE_THRESHOLD"]
)


# -----------------------------
# Data, scaler and models
# -----------------------------
@st.cache_data
def load_data():
    raw = pd.read_csv(root_path("data", "raw", "water_quality.csv"))
    # Same cleaning as data/preprocess.py
    raw = raw.drop_duplicates()
    raw = raw.fillna(raw.mean(numeric_only=True))
    cleaned = pd.read_csv(root_path("data", "processed", "cleaned_data.csv"))
    test = pd.read_csv(root_path("data", "processed", "test_data_global.csv"))
    return raw, cleaned, test


raw_df, df, test_df = load_data()
feature_cols = df.drop(columns=[TARGET_COL]).columns.tolist()
input_dim = len(feature_cols)
class_labels = sorted(df[TARGET_COL].unique().tolist())
num_classes = len(class_labels)


@st.cache_resource
def load_scaler():
    """Scaler used in preprocessing; rebuilt from raw data if the file is missing."""
    path = root_path("saved_models", "scaler.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    scaler = StandardScaler()
    scaler.fit(raw_df[feature_cols].values)
    return scaler


@st.cache_resource
def load_lr_rf_models():
    try:
        lr = joblib.load(root_path("saved_models", "logistic_regression.pkl"))
        rf = joblib.load(root_path("saved_models", "random_forest.pkl"))
        return lr, rf
    except FileNotFoundError:
        return None, None


@st.cache_resource
def load_model(input_dim, output_dim):
    model = WaterQualityModel(input_dim, output_dim=output_dim)
    model_path = root_path("global_model.pth")
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        except Exception:
            st.warning("Could not load model weights. Using initialized weights.")
    else:
        st.warning("Model file 'global_model.pth' not found. Using initialized weights.")
    model.eval()
    return model


scaler = load_scaler()
lr_model, rf_model = load_lr_rf_models()
model = load_model(input_dim, num_classes)


def entropy_confidence(prob_dist):
    eps = 1e-8
    entropy = -np.sum(prob_dist * np.log(prob_dist + eps))
    max_entropy = np.log(len(prob_dist))
    return float(1.0 - min(max(entropy / (max_entropy + eps), 0.0), 1.0))


# -----------------------------
# Input Section
# -----------------------------
st.subheader("📥 Enter Water Sample Parameters")
st.caption(
    "Enter raw measurements. Defaults are the dataset medians. "
    "Values are standardized with the training scaler before prediction."
)

input_data = {}

rows = 3
cols_per_row = (len(feature_cols) + rows - 1) // rows
cols = st.columns(cols_per_row)

for i, col in enumerate(feature_cols):
    with cols[i % cols_per_row]:
        input_data[col] = st.number_input(
            col, value=float(raw_df[col].median()), format="%.4f"
        )

input_raw = np.array([[input_data[c] for c in feature_cols]], dtype=float)
input_scaled = scaler.transform(pd.DataFrame(input_raw, columns=feature_cols))
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Analyze Water Sample"):
    if model_choice == "Logistic Regression":
        if lr_model is None:
            st.error("Logistic Regression model not found in 'saved_models/'. Please run training first.")
            st.stop()
        prob_dist = lr_model.predict_proba(input_scaled)[0]
        classes = list(lr_model.classes_)
        confidence = entropy_confidence(prob_dist)

    elif model_choice == "Random Forest":
        if rf_model is None:
            st.error("Random Forest model not found in 'saved_models/'. Please run training first.")
            st.stop()
        prob_dist = rf_model.predict_proba(input_scaled)[0]
        classes = list(rf_model.classes_)
        confidence = entropy_confidence(prob_dist)

    else:
        probs, confidence, _, _ = confidence_decision(
            model, input_tensor, threshold=confidence_threshold
        )
        prob_dist = np.array(probs[0])
        classes = class_labels

    pred_idx = int(np.argmax(prob_dist))
    pred_class = classes[pred_idx]
    decision = f"Water Quality Class {pred_class}"
    if confidence < confidence_threshold:
        decision = "⚠ Lab Testing Recommended"

    st.subheader("🧠 Prediction Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Status", decision)
    col2.metric("Confidence Score", f"{confidence:.3f}")
    col3.metric("Model Used", model_choice)

    st.markdown("**Class probabilities**")
    st.bar_chart(
        pd.DataFrame(
            {"Probability": prob_dist},
            index=[f"Class {c}" for c in classes]
        )
    )

    # -----------------------------
    # Confidence Gauge
    # -----------------------------
    st.subheader("🎯 Confidence Gauge")
    st.progress(min(max(confidence, 0.0), 1.0))
    st.caption("Low confidence → model recommends laboratory testing")

    # -----------------------------
    # Export Report
    # -----------------------------
    st.subheader("📄 Export Prediction Report")
    report = {"Model Used": model_choice, **input_data}
    for c, p in zip(classes, prob_dist):
        report[f"P(Class {c})"] = float(p)
    report["Predicted Class"] = pred_class
    report["Confidence Score"] = confidence
    report["Decision"] = decision
    report_df = pd.DataFrame([report])

    st.download_button(
        label="⬇ Download Report (CSV)",
        data=report_df.to_csv(index=False),
        file_name="water_quality_prediction.csv",
        mime="text/csv"
    )

# -----------------------------
# Visualization Section
# -----------------------------
st.subheader("📊 System Insights")


@st.cache_data
def load_experiment_logs():
    logs = []
    for path in sorted(glob.glob(root_path("experiments", "logs", "seed_*_log.json"))):
        with open(path) as f:
            logs.append(json.load(f))
    return logs


logs = load_experiment_logs()

with st.expander("Client Uncertainty Visualization", expanded=True):
    if logs:
        regions = list(logs[0]["uncertainties"][-1].keys())
        uncertainties = [
            float(np.mean([np.mean(log["uncertainties"][-1][r]) for log in logs]))
            for r in regions
        ]
        client_names = [r.capitalize() for r in regions]
        st.caption(f"Mean over {len(logs)} seeds, last training round (from experiments/logs).")
    else:
        client_names = ["Industrial", "Residential", "Agricultural"]
        uncertainties = [0.0, 0.0, 0.0]
        st.info("No experiment logs found. Run experiments/run_experiment.py first.")

    colors = ["red" if u > 0.2 else "green" for u in uncertainties]
    fig, ax = plt.subplots()
    ax.bar(client_names, uncertainties, color=colors)
    ax.set_ylabel("Normalized Uncertainty (Entropy)")
    ax.set_title("Region-wise Uncertainty Levels (Last Round)")
    st.pyplot(fig)
    st.caption("Red indicates high uncertainty (> 0.2), reducing client influence.")

    if logs:
        acc_df = pd.DataFrame(
            {f"Seed {log['seed']}": log["round_accuracies"] for log in logs}
        )
        acc_df.index = acc_df.index + 1
        acc_df.index.name = "Round"
        st.markdown("**Global test accuracy per FL round**")
        st.line_chart(acc_df)


@st.cache_data
def compute_importance():
    X_val = test_df.drop(columns=[TARGET_COL]).values
    y_val = test_df[TARGET_COL].values
    np.random.seed(0)
    return permutation_importance(model, X_val, y_val, feature_cols, n_repeats=3)


with st.expander("🔍 Explainability: Feature Importance"):
    st.markdown(
        "Permutation importance of the Hybrid FL model on the global test set: "
        "drop in accuracy when each parameter is shuffled."
    )
    importances = compute_importance()
    st.bar_chart(pd.Series(importances).sort_values(ascending=False))


@st.cache_data
def compute_drift():
    train_df = pd.read_csv(root_path("data", "processed", "train_data.csv"))
    reference = train_df.drop(columns=[TARGET_COL]).values
    rows_out = []
    for name in ["industrial", "residential", "agricultural"]:
        parts = [
            pd.read_csv(root_path("data", "splits", f"client_{name}_{i}.csv"))
            for i in (1, 2)
        ]
        current = pd.concat(parts).drop(columns=[TARGET_COL]).values
        drift, psi = detect_drift(reference, current, threshold=CONFIG["DRIFT_THRESHOLD"])
        rows_out.append({
            "Client Region": name.capitalize(),
            "PSI vs Train Pool": round(float(psi), 4),
            "Drift": "Yes" if drift else "No",
        })
    return pd.DataFrame(rows_out)


with st.expander("Concept Drift Status", expanded=True):
    drift_df = compute_drift()
    st.write(f"PSI threshold: `{CONFIG['DRIFT_THRESHOLD']}`")
    st.dataframe(drift_df, hide_index=True)
    if logs:
        events = sorted({e for log in logs for e in log["drift_events"]})
        status = f"Drift-triggered fine-tuning in rounds {events}" if events else "No drift detected during training"
        st.write(f"Training log status: **{status}**")
        checks = logs[0].get("drift_checks", [])
        drifted = [c for c in checks if c["drift"]]
        if drifted:
            st.markdown("**Fine-tuning triggers during training** (seed 0)")
            st.dataframe(
                pd.DataFrame(drifted)[["round", "region", "psi"]].rename(
                    columns={"round": "Round", "region": "Region", "psi": "PSI"}
                ).round(4),
                hide_index=True,
            )

with st.expander("📘 Training Summary"):
    st.markdown("""
    **Models Used**
    - Logistic Regression (Baseline)
    - Random Forest (Ensemble ML)
    - Proposed Hybrid Federated Learning (MLP)

    **Key Novelties**
    - Uncertainty-driven aggregation
    - Hierarchical federated learning
    - Concept drift–aware hybrid training
    - Confidence-aware decision support
    - Explainability via feature importance

    **Evaluation Protocol**
    - Global unseen test set
    - 5-seed experiments
    - Mean ± standard deviation reporting
    """)
