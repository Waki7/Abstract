import gym.spaces as gym_spaces
import numpy as np
import torch

import settings


def true_with_probability(p):
    return np.random.choice([True, False], 1, [p, 1 - p])


def sum_multi_modal_shapes(shapes):
    total_features = 0
    for shape in shapes:
        if isinstance(shapes, int):
            total_features += shape
        elif isinstance(shape, float):
            total_features += int(shape)
        elif isinstance(shape, tuple) or isinstance(shape, np.ndarray) or isinstance(shape, list):
            total_features += np.prod(shape)
        else:
            raise NotImplementedError('type of shape is not supported, feel free to add it.')
    return total_features


def spaces_to_shapes(spaces: gym_spaces.Space):
    shapes = []
    for space in spaces:
        if isinstance(space, gym_spaces.Discrete):
            shapes.append((space.n,))
        elif isinstance(space, gym_spaces.Box):
            shape = space.shape
            shapes.append(shape)
        else:
            raise NotImplementedError('have not implemented calculation for other spaces yet')
    return shapes


def get_target_action(n_actions, actions_taken, advantage):
    signs = (torch.sign(advantage) - 1) / 2

    target_action = torch.zeros(len(actions_taken), n_actions).to(settings.DEVICE)
    actions_taken = torch.tensor(actions_taken).to(settings.DEVICE).unsqueeze(-1)
    target_action.scatter_(dim=1, index=actions_taken, value=1)
    signs = signs.unsqueeze(-1).repeat(1, n_actions)
    # scaling = torch.abs(advantage.unsqueeze(-1).repeat(1, n_actions))
    target_action = torch.abs(target_action + signs)

    return target_action


def convert_env_batch_input(env_inputs, action=None):
    env_inputs = torch.from_numpy(env_inputs).to(settings.DEVICE).float().unsqueeze(0)
    if action is not None:
        return torch.cat([env_inputs, action], dim=-1)
    return env_inputs


def one_hot(logits, idx=None):
    max_idx = torch.argmax(logits, dim=-1, keepdim=True) if idx is None else idx
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(-1, max_idx, 1)
    return one_hot


# ---------------------------------------------------------------------------
# numeric property utilities
# ---------------------------------------------------------------------------
def is_odd(val):
    return val % 2 == 1