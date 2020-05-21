import gym
import numpy as np


class NaivePolicy():
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Discrete):
        self.output = np.zeros((action_space.n))
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_actions = self.action_space.n

    def step(self, obs, **kwargs):
        pass


class RandomPolicy(NaivePolicy):
    # don't need obs
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Discrete):
        super(RandomPolicy, self).__init__(observation_space, action_space)
        self.random = np.arange(len(self.output))

    def step(self, **kwargs):
        return np.random.randint(low=0, high=self.n_actions - 1)


class FollowPolicy(NaivePolicy):
    # need one coordinate obs
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Discrete):
        super(FollowPolicy, self).__init__(observation_space, action_space)


class AvoidPolicy(NaivePolicy):
    # need one cooridnate obs
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Discrete):
        super(AvoidPolicy, self).__init__(observation_space, action_space)
