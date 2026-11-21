import torch

def fedavg(state_dicts):
    avg_state = {}

    for key in state_dicts[0].keys():
        # Stack all tensors for this key
        # Check if they are float tensors -> average
        # If they are Long tensors (like num_batches_tracked), casting to float might be needed or just take one.
        # Usually model weights are float.
        avg_state[key] = torch.mean(
            torch.stack([sd[key].float() for sd in state_dicts]),
            dim=0
        )

    return avg_state
