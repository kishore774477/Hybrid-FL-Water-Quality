# Hybrid FL Water Quality

A **Hybrid Federated Learning** system for water quality monitoring. Simulated clients (industrial, residential and agricultural regions) train a shared neural network without sharing raw data. The system combines:

- **Uncertainty-weighted aggregation**: clients with lower prediction entropy get more influence.
- **Hierarchical FL**: client models are aggregated within each region first, then across regions.
- **Concept drift detection (PSI)**: every 5 rounds, each region's data is compared with the pooled training distribution. A region whose PSI is above `DRIFT_THRESHOLD` triggers centralized fine-tuning on that region's data. An equal-sized replay sample of the reference data is mixed in, so the model doesn't forget the other regions.
- **Confidence-aware decisions**: low-confidence predictions are flagged for lab testing.
- **Explainability**: permutation feature importance.
- **Baselines**: Logistic Regression and Random Forest, trained on the same client data.

The dataset (`data/raw/water_quality.csv`) has 14 physico-chemical parameters and a 3-class `Water Quality` label (0, 1, 2).

## 🚀 Quick Start

Requires Python 3.10+. Run all commands from the project folder.

### 1. Create the virtual environment (first time only)

**Windows (PowerShell)**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

> If PowerShell says *running scripts is disabled*, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

### 2. Run the dashboard
The repository already contains the trained models, so you can start the dashboard right away:

```powershell
.\.venv\Scripts\Activate.ps1      # Windows (skip if (.venv) is already shown)
python -m streamlit run gui/dashboard.py
```

Then open **http://localhost:8501**. Keep the terminal open while you use the dashboard; press `Ctrl+C` to stop it.

Enter raw water measurements (the defaults are the dataset medians), choose a model and click **Analyze Water Sample**. Inputs are standardized with the training scaler (`saved_models/scaler.pkl`) before prediction.

### Troubleshooting

**`Fatal error in launcher: Unable to create process using ...python.exe`**
The `.venv` was created in a different folder and the project was moved (for example, into OneDrive). A virtual environment can't be moved, so rebuild it:

```powershell
deactivate                          # if (.venv) is shown
Remove-Item .venv -Recurse -Force
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Always start the dashboard with `python -m streamlit ...` rather than `streamlit ...`; it keeps working even if the launcher `.exe` files break.

**`localhost refused to connect` / `ERR_CONNECTION_REFUSED`**
The dashboard isn't running. Start it with the command in step 2 and keep that terminal open.

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
| Hybrid FL (mean over seeds) | 93.54% ± 0.54% |
| Hybrid FL (best saved model) | 94.19% |

In each run the industrial (PSI ≈ 0.80) and agricultural (PSI ≈ 0.61) regions trigger drift fine-tuning at rounds 5, 10 and 15. The residential region (PSI ≈ 0.03) does not.

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
