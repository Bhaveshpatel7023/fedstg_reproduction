import torch

def fedAvg(state_dicts, client_sizes):
    total_samples = sum(client_sizes)

    avg_state = {}

    for key in state_dicts[0]:
        avg_state[key] = sum(
            state[key].float() * (size / total_samples)
            for state, size in zip(state_dicts, client_sizes)
        )

    return avg_state