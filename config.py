"""
Configuration for Hybrid FL Water Quality Monitoring.
Contains hyperparameters and global constants to ensure consistency
across all experiments and deployments.
"""

CONFIG = {
    "SEEDS": [0, 1, 2, 3, 4],
    "ROUNDS": 15,
    "LOCAL_EPOCHS": 3,
    "CONFIDENCE_THRESHOLD": 0.7,
    "TARGET_COL": "Water Quality",
    "DRIFT_THRESHOLD": 0.2
}
