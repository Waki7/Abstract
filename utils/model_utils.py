import numpy as np
import torch

import settings


def true_with_probability(p):
    return np.random.choice([True, False], 1, [p, 1 - p])


def get_target_action(n_actions, actions_taken, advantage):
    signs = (torch.sign(advantage)-1) / 2

    target_action = torch.zeros(len(actions_taken), n_actions).to(settings.DEVICE)
    actions_taken = torch.tensor(actions_taken).to(settings.DEVICE).unsqueeze(-1)
    target_action.scatter_(dim=1, index=actions_taken, value=1)
    signs = signs.unsqueeze(-1).repeat(1, n_actions)
    # scaling = torch.abs(advantage.unsqueeze(-1).repeat(1, n_actions))
    target_action = torch.abs(target_action + signs)

    return target_action


def convert_env_input(env_input):
    return torch.from_numpy(env_input).to(settings.DEVICE).float().unsqueeze(0)


def one_hot(logits, idx=None):
    max_idx = torch.argmax(logits, dim=-1, keepdim=True) if idx is None else idx
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(-1, max_idx, 1)
    return one_hot
