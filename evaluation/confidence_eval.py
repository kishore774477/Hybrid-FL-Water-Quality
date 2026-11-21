import torch
import torch.nn.functional as F
import numpy as np

def confidence_decision(model, X, threshold=0.7):
    """
    Evaluates confidence of the prediction based on entropy.
    Adapted for Multi-Class Classification.
    X: Expects a single sample (batch dimension 1) or handling it as such.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1)
        
    eps = 1e-8
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    
    num_classes = probs.shape[1]
    max_entropy = np.log(num_classes)
    
    normalized_entropy = (entropy.mean().item()) / (max_entropy + eps)
    confidence = 1.0 - min(max(normalized_entropy, 0.0), 1.0)
    
    _, pred_idx = torch.max(probs, dim=1)
    pred_class = pred_idx.item()
    
    if confidence < threshold:
        decision = "⚠ Lab Testing Recommended"
    else:
        classes = {0: "Water Quality Class 0", 1: "Water Quality Class 1", 2: "Water Quality Class 2"} 
        decision = f"Confident: {classes.get(pred_class, 'Unknown')}"

    return probs.tolist(), confidence, decision, pred_class
