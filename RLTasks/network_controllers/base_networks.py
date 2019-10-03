import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import RLTasks.config as cfg
import numpy as np
from utils.NetworkTypes import *
import gym

grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook


class PolicyNetworkBasic(nn.Module):
    def __init__(self, env: gym.Env):
        super(PolicyNetworkBasic, self).__init__()
        if isinstance(env.observation_space, gym.spaces.Box):
            num_inputs = env.observation_space.shape[0]
        else:
            if isinstance(env.observation_space, gym.spaces.Discrete):
                num_inputs = env.observation_space.n
            else:
                raise NotImplementedError
        num_actions = env.action_space.n
        assert isinstance(env.action_space, gym.spaces.Discrete)

        hidden_size = cfg.gym.hidden_size
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.train()


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        probs = self.forward(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = -torch.log(probs.squeeze(0)[highest_prob_action])  # same as nlloss(log(softmax(x))
        return highest_prob_action, log_prob


class ACNetwork(nn.Module):  # actor critic method, parameterized baseline estimate with network
    def __init__(self, env: gym.Env):
        super(ACNetwork, self).__init__()
        if isinstance(env.observation_space, gym.spaces.Box):
            num_inputs = env.observation_space.shape[0]
        else:
            if isinstance(env.observation_space, gym.spaces.Discrete):
                num_inputs = env.observation_space.n
            else:
                raise NotImplementedError
        num_actions = env.action_space.n
        assert isinstance(env.action_space, gym.spaces.Discrete)

        hidden_size = cfg.gym.hidden_size
        self.num_actions = num_actions

        self.actor = nn.Sequential(nn.Linear(num_inputs, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size),
                                   nn.Tanh())

        self.critic = nn.Sequential(nn.Linear(num_inputs, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size),
                                    nn.Tanh())
        self.critic_linear = nn.Linear(hidden_size, 1)
        self.train()

    def forward(self, inputs, masks=1):
        x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), F.softmax(hidden_actor, dim=1)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value, probs = self.forward(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = -torch.log(probs.squeeze(0)[highest_prob_action])  # same as nlloss(log(softmax(x))
        return highest_prob_action, log_prob, value
