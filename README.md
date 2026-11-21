# Hybrid FL Water Quality

A **Hybrid Federated Learning** system for water quality monitoring. Simulated clients (industrial, residential and agricultural regions) train a shared neural network without sharing raw data. The system combines:

- **Uncertainty-weighted aggregation**: clients with lower prediction entropy get more influence.
- **Hierarchical FL**: client models are aggregated within each region first, then across regions.
- **Concept drift detection (PSI)**: when drift is detected, the global model is fine-tuned centrally.
- **Confidence-aware decisions**: low-confidence predictions are flagged for lab testing.
- **Explainability**: permutation feature importance.
- **Baselines**: Logistic Regression and Random Forest, trained on the same client data.

The dataset (`data/raw/water_quality.csv`) has 14 physico-chemical parameters and a 3-class `Water Quality` label (0, 1, 2).

## 🚀 Quick Start

### 1. Install dependencies
Requires Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the dashboard
The repository already contains the trained models, so you can start the dashboard right away:

```bash
streamlit run gui/dashboard.py
```

Enter raw water measurements (the defaults are the dataset medians), choose a model and click **Analyze Water Sample**. Inputs are standardized with the training scaler (`saved_models/scaler.pkl`) before prediction.

## 🔁 Reproduce the pipeline

Run the steps in this order. Each script works from any directory.

```bash
python data/preprocess.py               # clean + scale data, 80/20 train/test split, saves scaler
python clients/client.py                # build non-IID client splits (2 sub-clients per region)
python experiments/run_experiment.py    # hybrid FL, 5 seeds x 15 rounds -> global_model.pth + logs
python models_baseline/run_comparison.py # train LR / RF baselines, compare, feature importance
```

Hyperparameters (seeds, rounds, local epochs, thresholds) are set in `config.py`.

### Results (global held-out test set, 5 seeds)

| Model | Accuracy |
|---|---|
| Logistic Regression | 80.88% ± 0.33% |
| Random Forest | 99.05% ± 0.14% |
| Hybrid FL (mean over seeds) | 86.76% ± 2.57% |
| Hybrid FL (best saved model) | 89.20% |

## 📂 Project Structure

```
aggregation/       FedAvg, uncertainty-weighted and hierarchical aggregation
clients/           client data splits, local training, uncertainty estimation
data/              raw data, processed train/test sets, client splits
drift_detection/   Population Stability Index (PSI) drift detector
evaluation/        metrics and confidence-based decision logic
experiments/       main FL experiment + per-seed JSON logs
explainability/    permutation feature importance
gui/               Streamlit dashboard
hybrid/            drift-triggered centralized fine-tuning controller
model/             MLP model (WaterQualityModel)
models_baseline/   Logistic Regression / Random Forest baselines and comparison
saved_models/      trained baselines + feature scaler
global_model.pth   trained global Hybrid FL model
config.py          experiment configuration
```
