from typing import List, Union

import gym.spaces as gym_spaces
import numpy as np
import torch

import settings


# ---------------------------------------------------------------------------
# SPACES AND SHAPES
# ---------------------------------------------------------------------------

def true_with_probability(p):
    return np.random.choice([True, False], 1, [p, 1 - p])


# ---------------------------------------------------------------------------
# SPACES AND SHAPES
# ---------------------------------------------------------------------------
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


def batch_env_observations(observation_list: List[np.ndarray], space: gym_spaces.Space):
    if isinstance(space, gym_spaces.Tuple):
        batched_observation = []
        for obs_idx in range(len(observation_list[0])):
            obss = torch.stack([torch.tensor(obs_batch[obs_idx]) for obs_batch in observation_list])
            batched_observation.append(obss)
    elif isinstance(space, gym_spaces.Box):
        batched_observation = torch.stack([torch.tensor(obs_batch) for obs_batch in observation_list])
    else:
        raise NotImplementedError
    return batched_observation


def convert_env_batch_input(env_inputs: Union[List[torch.Tensor], torch.Tensor],
                            space: gym_spaces.Tuple, action: torch.Tensor = None):
    if isinstance(space, gym_spaces.Tuple):
        # treating as multimodal input
        env_inputs = [tensor.to(settings.DEVICE).float() for tensor in env_inputs]
        if action is not None:
            # TODO remove
            assert action.shape[0] == env_inputs[0].shape[0]
            env_inputs.append(action)
    else:
        # treating as unimodal input
        env_inputs = env_inputs.to(settings.DEVICE).float()
        if action is not None:
            env_inputs = torch.cat([env_inputs, action], dim=-1)
    return env_inputs


def get_target_action(n_actions, actions_taken, advantage):
    signs = (torch.sign(advantage) - 1) / 2

    target_action = torch.zeros(len(actions_taken), n_actions).to(settings.DEVICE)
    actions_taken = torch.tensor(actions_taken).to(settings.DEVICE).unsqueeze(-1)
    target_action.scatter_(dim=1, index=actions_taken, value=1)
    signs = signs.unsqueeze(-1).repeat(1, n_actions)
    # scaling = torch.abs(advantage.unsqueeze(-1).repeat(1, n_actions))
    target_action = torch.abs(target_action + signs)

    return target_action


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
