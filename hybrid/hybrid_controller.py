from drift_detection.drift_detector import detect_drift
from hybrid.central_finetune import centralized_finetune
from config import CONFIG

def hybrid_control(global_model, ref_df, curr_df, target_col):
    """
    Monitors drift between reference and current data.
    If drift > threshold, triggers centralized fine-tuning.
    Returns: (global_model, drift_flag)
    """
    # Keep as 2D array for feature-wise comparison
    ref_vals = ref_df.drop(columns=[target_col]).values
    curr_vals = curr_df.drop(columns=[target_col]).values
    
    drift_threshold = CONFIG.get("DRIFT_THRESHOLD", 0.2)
    drift, psi = detect_drift(ref_vals, curr_vals, threshold=drift_threshold)

    if drift:
        print(f"[!] Concept DRIFT DETECTED (PSI={psi:.3f}). Switching to Centralized Fine-Tuning.")
        try:
            global_model = centralized_finetune(
                global_model, curr_df, target_col
            )
            print("  -> Fine-tuning completed.")
        except Exception as e:
            print(f"  -> Fine-tuning failed: {e}")
    else:
        print(f"[OK] No significant drift (PSI={psi:.3f}). Continuing Federated Learning.")

    return global_model, drift
