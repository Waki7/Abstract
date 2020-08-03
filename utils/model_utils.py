import typing as typ

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
    prev_reward = torch.zeros(rewards.shape[-1]).to(**settings.ARGS)
    discounted_reward = torch.zeros_like(rewards).to(**settings.ARGS)
    for episode_idx in range(rewards.shape[0] - 1, -1, -1):
        discounted_reward[episode_idx] = rewards[episode_idx] + (prev_reward * discount)
        prev_reward = discounted_reward[episode_idx]
    return discounted_reward


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
def get_activation_for_space(space: gym_spaces.Space, in_features):
    out_features = sum_multi_modal_shapes(space_to_shapes(space))
    if isinstance(space, gym_spaces.Tuple):
        assert all([isinstance(space, type(space[0])) for space in space])
        space = space.spaces[0]
    if isinstance(space, gym_spaces.Discrete):
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features),
            torch.nn.Softmax()
        )
    if isinstance(space, gym_spaces.Box):
        return torch.nn.Linear(in_features=in_features, out_features=out_features)


def sum_multi_modal_shapes(shapes):
    total_features = 0
    for shape in shapes:
        if isinstance(shape, int):
            total_features += shape
        elif isinstance(shape, float):
            total_features += int(shape)
        elif isinstance(shape, tuple) or isinstance(shape, np.ndarray) or isinstance(shape, list):
            total_features += np.prod(shape)
        else:
            raise NotImplementedError('type of shape {} is not supported, feel free to add it.'.format(type(shape)))
    return total_features


def space_to_shapes(space: gym_spaces.Space) -> typ.Union[typ.Tuple, typ.List[typ.Tuple]]:
    def space_to_shape(space):
        if isinstance(space, gym_spaces.Discrete):
            return space.n,
        elif isinstance(space, gym_spaces.Box):
            return space.shape
        else:
            raise NotImplementedError('have not implemented calculation for other spaces yet')

    if isinstance(space, gym_spaces.Tuple):
        shapes = [space_to_shape(spce) for spce in space.spaces]
        return shapes
    else:
        return [space_to_shape(space)]


def scale_space(state: np.ndarray, space):
    if isinstance(space, gym_spaces.Tuple):
        return [scale_space(i_state, i_space) for i_state, i_space in zip(state, space.spaces)]
    if isinstance(space, gym_spaces.Discrete):
        state = state / space.n
    elif isinstance(space, gym_spaces.Box):
        if space.low.shape[-1] == state.shape[-1]:
            state = (state - space.low) / (space.high - space.low)
        else:
            low = np.expand_dims(space.low.flatten(), axis=0)
            high = np.expand_dims(space.high.flatten(), axis=0)
            state = (state - low) / (high - low)
    else:
        raise NotImplementedError('have not implemented calculation for other spaces yet')
    return state


# ---------------------------------------------------------------------------
# DATA CONVERSION
# ---------------------------------------------------------------------------

def batch_env_observations(observation_list: typ.List[np.ndarray], space: gym_spaces.Space) -> typ.Union[
    typ.List[np.ndarray], np.ndarray]:
    if isinstance(space, gym_spaces.Tuple):
        batched_observation = []
        for obs_idx in range(len(observation_list[0])):
            obss = np.stack([obs_batch[obs_idx] for obs_batch in observation_list])
            batched_observation.append(obss)
    elif isinstance(space, gym_spaces.Box):
        batched_observation = np.stack([obs_batch for obs_batch in observation_list])
    else:
        raise NotImplementedError
    return batched_observation


def list_to_torch_device(env_inputs: typ.Union[typ.List[torch.Tensor], torch.Tensor]):
    # treating as multimodal input
    env_inputs = [torch.tensor(tensor).to(**settings.ARGS) for tensor in env_inputs]
    return env_inputs


def get_idxs_of_list(list, idxs):
    return [list[i] for i in idxs]


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


def scale_vector_to_range(vector: typ.Union[np.ndarray, torch.Tensor], new_min: typ.Union[int, float],
                          new_max: [int, float]):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(vector), np.max(vector)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * vector + b

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
