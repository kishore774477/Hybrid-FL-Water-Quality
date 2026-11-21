import torch
from evaluation.metrics import evaluate_model

def evaluate_proposed_model(model, X_test, y_test):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    metrics = evaluate_model(model, X_test, y_test)
    return metrics["accuracy"]
