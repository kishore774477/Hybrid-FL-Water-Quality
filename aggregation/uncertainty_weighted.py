import torch

def uncertainty_weighted_aggregation(state_dicts, uncertainties):
    """
    state_dicts: list of model weight dictionaries
    uncertainties: list of float uncertainty values (normalized to [0, 1])
    """

    # Weight = Confidence = 1 - Uncertainty
    weights = [1 - u for u in uncertainties]
    
    # Handle edge case where all weights are 0 (all uncertainties are 1)
    if sum(weights) == 0:
        weights = [1.0] * len(weights)
        
    total_weight = sum(weights)

    normalized_weights = [w / total_weight for w in weights]

    agg_state = {}

    for key in state_dicts[0].keys():
        # Ensure floating point calculation
        current_layer_weights = []
        for i in range(len(state_dicts)):
            w = normalized_weights[i]
            param = state_dicts[i][key].float()
            current_layer_weights.append(w * param)
            
        agg_state[key] = sum(current_layer_weights)

    return agg_state, normalized_weights
