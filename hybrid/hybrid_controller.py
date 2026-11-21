from drift_detection.drift_detector import detect_drift
from hybrid.central_finetune import centralized_finetune
from config import CONFIG
import pandas as pd

def hybrid_control(global_model, ref_df, curr_df, target_col, region=None):
    """
    Monitors drift between reference and current data.
    If drift > threshold, triggers centralized fine-tuning on the drifted data.
    To avoid forgetting the other regions, the drifted data is mixed with an
    equal-sized replay sample of the reference data (rehearsal).
    Returns: (global_model, drift_flag, psi)
    """
    label = f"[{region}] " if region else ""
    # Keep as 2D array for feature-wise comparison
    ref_vals = ref_df.drop(columns=[target_col]).values
    curr_vals = curr_df.drop(columns=[target_col]).values
    
    drift_threshold = CONFIG.get("DRIFT_THRESHOLD", 0.2)
    drift, psi = detect_drift(ref_vals, curr_vals, threshold=drift_threshold)

    if drift:
        print(f"{label}[!] Concept DRIFT DETECTED (PSI={psi:.3f}). Switching to Centralized Fine-Tuning.")
        try:
            replay = ref_df.sample(
                n=min(len(curr_df), len(ref_df)), random_state=CONFIG.get("REPLAY_SEED", 0)
            )
            finetune_df = pd.concat([curr_df, replay], ignore_index=True)
            global_model = centralized_finetune(
                global_model, finetune_df, target_col
            )
            print("  -> Fine-tuning completed.")
        except Exception as e:
            print(f"  -> Fine-tuning failed: {e}")
    else:
        print(f"{label}[OK] No significant drift (PSI={psi:.3f}). Continuing Federated Learning.")

    return global_model, bool(drift), float(psi)
