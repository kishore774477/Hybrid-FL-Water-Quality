import numpy as np

def population_stability_index(expected, actual, bins=10):
    """
    Calculate PSI using Histogram method.
    expected: Reference distribution (e.g. initial training data) 1D array
    actual: Current distribution (e.g. new client data) 1D array
    """
    expected = np.array(expected)
    actual = np.array(actual)

    # Define bins based on the REFERENCE (expected) distribution
    # We use global min/max to ensure coverage
    min_val = min(np.min(expected), np.min(actual))
    max_val = max(np.max(expected), np.max(actual))
    
    # Handle case where all values are same
    if min_val == max_val:
        return 0.0

    breakpoints = np.linspace(min_val, max_val, bins + 1)
    
    # Calculate frequencies
    expected_cnts, _ = np.histogram(expected, bins=breakpoints)
    actual_cnts, _ = np.histogram(actual, bins=breakpoints)

    # Normalize to probabilities (fractions)
    # Add epsilon to prevent div by zero
    expected_percs = expected_cnts / len(expected) + 1e-6
    actual_percs = actual_cnts / len(actual) + 1e-6

    # PSI Formula
    psi_values = (actual_percs - expected_percs) * np.log(actual_percs / expected_percs)
    psi = np.sum(psi_values)

    return psi

def detect_drift(reference_data, current_data, threshold=0.2):
    """
    reference_data: 2D array of feature values (samples x features)
    current_data: 2D array of new values (samples x features)
    Computes PSI per feature and averages them.
    """
    ref_arr = np.array(reference_data)
    curr_arr = np.array(current_data)
    
    if len(ref_arr.shape) == 1:
        # Fallback if 1D array is passed
        psi = population_stability_index(ref_arr, curr_arr)
    else:
        psi_scores = []
        for i in range(ref_arr.shape[1]):
            psi_curr = population_stability_index(ref_arr[:, i], curr_arr[:, i])
            psi_scores.append(psi_curr)
        psi = np.mean(psi_scores)
        
    return psi > threshold, psi
