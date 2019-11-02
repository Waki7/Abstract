












from ray.rllib.policy.policy import Policy
import gym
import agent_models
import torch
import settings
from enum import Enum
import numpy as np
from os.path import isfile
import os
from utils.storage_utils import save_object, load_object


class FillVals(Enum):  # Filled values for grid, indicate the integer and what it represents
    Unseen = 0
    Seen = 1  # if self has explored/seen a point
    Occupied = 2  # if self is currently in a point
    OtherSeen = -1  # if another agent has explored/seen a point
    OtherOccupied = -2  # if another agent is currently at a point


class Transition():
    def __init__(self, state, action, reward, new_state, episode_end):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.episode_end = episode_end


ACTIONS = [
    # mostly copy of environment actions, just way to quickly fill surroundings for field of view
    # since the grid representation is H x W, the tuple values indicate change in direction (dy, dx) or (dH, dW),
    (0, -1),  # Left = 0
    (-1, -1),  # UpLeft = 1
    (-1, 0),  # Up = 2
    (-1, 1),  # UpRight = 3
    (0, 1),  # Right = 4
    (1, 1),  # DownRight = 5
    (1, 0),  # Down = 6
    (1, -1),  # DownLeft = 7
]


class BaseAgent():
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces, cfg):
        Policy.__init__(self, observation_space, action_space, cfg)
        self.cfg = cfg
        assert isinstance(observation_space, gym.spaces.Dict)
        assert isinstance(observation_space.spaces.get('position'), gym.spaces.Box), \
            'grid agent_algorithms currently implemented for observed dict including a box space for key \'position\''
        assert len(observation_space.spaces['position'].shape) > 1, \
            'expecting n+1 coordinates, own coordinates + n_agent coordinates'
        self.grid_shape = observation_space.spaces['position'].high[0]
        self.grid = np.zeros(self.grid_shape)
        self.positions = None
        self.num_actions = action_space.n
        self.t = 0
        self.state = None
        self.training = True
        self.discount_factor = self.cfg.get('discount_factor', settings.defaults.DISCOUNT_FACTOR)

    def process_env_input(self, env_input: dict):
        assert isinstance(env_input, dict)
        positions = env_input['position']
        assert isinstance(env_input['position'][0], tuple), 'expecting list of tuples for key \'position\''
        fovs = env_input.get('fov')

        if fovs is not None:
            for view_points in fovs[1:]:
                for view_point in view_points:
                    self.grid[view_point] = FillVals.OtherSeen.value
            for view_points in fovs[0]:  # for the fov points for current agent
                for view_point in view_points:
                    self.grid[view_point] = FillVals.Seen.value

        # if we have filled in previously, we want to change the old values to seen instead of currently occupied
        if self.positions is not None:
            for all_old_agents in self.positions[1:]:
                self.grid[all_old_agents] = FillVals.OtherSeen.value
            self.grid[self.positions[0]] = FillVals.Seen.value

        # the first env_input is expected to be the coordinates of the current agent
        for all_agents in positions[1:]:
            self.grid[all_agents] = FillVals.OtherOccupied.value
        self.grid[positions[0]] = FillVals.Occupied.value

        self.positions = positions
        return np.expand_dims(self.grid, axis=0)

    def end_episode(self):
        self.t = 0
        self.grid = np.zeros(self.grid_shape)
        self.rewards = []
        self.probs = []
        self.value_estimates = []

    def train(self):
        self.training = True

    def test(self):
        self.training = False

    def save(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        save_object(self, dir)


class BaseQLearner(BaseAgent):
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces, cfg):
        super(BaseQLearner, self).__init__(observation_space, action_space, cfg)
        self.epsilon_start = cfg.get('epsilon_start', settings.defaults.Q_EPSILON_START)
        self.epsilon_end = cfg.get('epsilon_end', settings.defaults.Q_EPSILON_END)
        self.epsilon_decay_length = cfg.get('epsilon_decay_length', settings.defaults.Q_EPSILON_DECAY_LENGTH)
        self.batch_size = cfg.get('batch_size', settings.defaults.Q_BATCH_SIZE)
        self.replay_capacity = cfg.get('replay_capacity', settings.defaults.Q_REPLAY_CAPACITY)
        self.target_reset_freq = cfg.get('target_reset_freq', settings.defaults.Q_TARGET_RESET_FREQ)
        self.replay_memory = [None] * self.replay_capacity

    def sample_transitions(self):
        transitions = np.random.choice(self.replay_memory, self.batch_size, replace=True)
        q_targets = []
        q_preds = []
        for transition in transitions:
            if transition is not None:
                q_preds.append(self.qn.get_q_estimate(transition.state))
                if transition.episode_end:
                    q_targets.append(transition.reward)
                else:
                    q_targets.append(transition.reward + self.discount_factor * np.argmax(
                        self.qn.get_q_estimate(transition.new_state).detach().cpu().numpy()))
                # expanding the dimensions and converting to pytorch tensor
                q_targets[-1] = torch.autograd.Variable(
                    torch.Tensor([q_targets[-1]]).to(**settings.ARGS))
        return q_preds, q_targets

    def get_epsilon(self):
        return (self.epsilon_start + ((self.epsilon_end - self.epsilon_start) / self.epsilon_decay_length) *
                self.t) if self.t < self.epsilon_decay_length else self.epsilon_end


class BasePGLearner(BaseAgent):
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces, cfg):
        super(BasePGLearner, self).__init__(observation_space, action_space, cfg)
        self.probs = []
        self.rewards = []
        self.value_estimates = []
