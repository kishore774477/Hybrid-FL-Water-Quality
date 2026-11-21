import numpy as np
import torch
from sklearn.metrics import accuracy_score

def permutation_importance(model, X, y, feature_names, n_repeats=5):
    """
    Computes permutation feature importance for a trained PyTorch model
    """
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Baseline accuracy
    with torch.no_grad():
        logits = model(X_tensor)
        baseline_preds = torch.argmax(logits, dim=1).numpy()
    
    baseline_acc = accuracy_score(y, baseline_preds)

    importances = {}

    for i, feature in enumerate(feature_names):
        scores = []

        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])

            Xp_tensor = torch.tensor(X_permuted, dtype=torch.float32)
            with torch.no_grad():
                logits_p = model(Xp_tensor)
                preds = torch.argmax(logits_p, dim=1).numpy()

            acc = accuracy_score(y, preds)
            scores.append(baseline_acc - acc)

        importances[feature] = np.mean(scores)

    return importances
