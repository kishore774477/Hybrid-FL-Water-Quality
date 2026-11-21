from aggregation.uncertainty_weighted import uncertainty_weighted_aggregation

def hierarchical_aggregation(region_models):
    """
    region_models: dict
    {
        "industrial": (state_dicts, uncertainties),
        "residential": (state_dicts, uncertainties),
        ...
    }
    """

    regional_states = []
    regional_uncertainties = []

    # 1. Intra-Region Aggregation
    for region, (states, uncertainties) in region_models.items():
        if not states: continue
        
        region_state, _ = uncertainty_weighted_aggregation(
            states, uncertainties
        )
        regional_states.append(region_state)
        
        # Average uncertainty of the region
        regional_uncertainties.append(sum(uncertainties) / len(uncertainties))

    # 2. Inter-Region (Global) Aggregation
    global_state, normalized_weights = uncertainty_weighted_aggregation(
        regional_states, regional_uncertainties
    )

    return global_state
