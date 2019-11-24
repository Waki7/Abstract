import numpy as np
import settings
import torch


def true_with_probability(p):
    return np.random.choice([True, False], 1, [p, 1 - p])


def convert_env_input(env_input):
    return torch.from_numpy(env_input).to(settings.DEVICE).float().unsqueeze(0)

def one_hot(logits):
    max_idx = torch.argmax(logits, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(-1, max_idx, 1)
    return one_hot