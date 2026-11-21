import torch
import torch.nn.functional as F
import numpy as np

def compute_uncertainty(model, X, eps=1e-8):
    """
    Uncertainty = average entropy of predictions.
    Adapted for Multi-Class Classification.
    Normalized to [0, 1] range to work with (1 - u) weighting logic.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1)

    # Multi-Class Entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    
    avg_entropy = entropy.mean().item()
    
    # Normalize by log(num_classes) so max entropy is 1.0
    # This prepares it for the aggregation logic: weight = 1 - uncertainty
    num_classes = probs.shape[1]
    max_entropy = np.log(num_classes)
    
    normalized_entropy = avg_entropy / (max_entropy + eps)
    
    # Clamp just in case
    return min(max(normalized_entropy, 0.0), 1.0)
