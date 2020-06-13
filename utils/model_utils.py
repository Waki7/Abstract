from typing import List, Union

import gym.spaces as gym_spaces
import numpy as np
import torch

import settings


# ---------------------------------------------------------------------------
# FUNCTIONS/ALGORITHMS
# ---------------------------------------------------------------------------

def discount_rewards(rewards: torch.Tensor, discount: float, td_step: int = -1) -> torch.Tensor:
    if td_step != -1:
        raise NotImplementedError('currently only implemented monte carlo esimation')
    prev_reward = torch.zeros(rewards.shape[-1])
    discounted_reward = torch.zeros_like(rewards)
    for episode_idx in range(rewards.shape[0] - 1, -1, -1):
        discounted_reward[episode_idx] = rewards[episode_idx] + (prev_reward * discount)
        prev_reward = rewards[episode_idx]
    return discounted_reward.to(settings.DEVICE)


# ---------------------------------------------------------------------------
# PROBABILITY AND SAMPLING
# ---------------------------------------------------------------------------

def true_with_probability(p):
    return np.random.choice([True, False], 1, [p, 1 - p])


def random_choice_prob_batch(n: int, batch_ps: np.ndarray) -> np.ndarray:
    '''
    perform numpy random select over a batch of probabilities
    :param n: number of indices to select from for each batch
    :param batch_ps: b x n
    :param size: optional, select this number of elements from each batch
    :return: a numpy array A (selected) of shape b x size where for all a in A 0 >= x < n
    '''
    selected_idxs = np.zeros(batch_ps.shape[0])
    for batch_idx, probs in enumerate(batch_ps):
        idx = np.random.choice(n, p=probs)
        selected_idxs[batch_idx] = idx
    return np.asarray(selected_idxs, dtype=np.long)


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


# ---------------------------------------------------------------------------
# DATA CONVERSION
# ---------------------------------------------------------------------------

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


def list_to_torch_device(env_inputs: Union[List[torch.Tensor], torch.Tensor]):
    # treating as multimodal input
    env_inputs = [tensor.to(settings.DEVICE).float() for tensor in env_inputs]
    return env_inputs


# ---------------------------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------------------------

def get_euclidean_distance(point1: np.ndarray, point2: np.ndarray):
    return np.linalg.norm(point1 - point2, ord=2)


# ---------------------------------------------------------------------------
# NUMERIC PROPERTIES
# ---------------------------------------------------------------------------
def is_odd(val):
    return val % 2 == 1

# def get_target_action(n_actions, actions_taken, advantage):
#     signs = (torch.sign(advantage) - 1) / 2
#
#     target_action = torch.zeros(len(actions_taken), n_actions).to(settings.DEVICE)
#     actions_taken = torch.tensor(actions_taken).to(settings.DEVICE).unsqueeze(-1)
#     target_action.scatter_(dim=1, index=actions_taken, value=1)
#     signs = signs.unsqueeze(-1).repeat(1, n_actions)
#     # scaling = torch.abs(advantage.unsqueeze(-1).repeat(1, n_actions))
#     target_action = torch.abs(target_action + signs)
#
#     return target_action
#
#
# def one_hot(logits, idx=None):
#     max_idx = torch.argmax(logits, dim=-1, keepdim=True) if idx is None else idx
#     one_hot = torch.zeros_like(logits)
#     one_hot.scatter_(-1, max_idx, 1)
#     return one_hot
