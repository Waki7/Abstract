import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import RLTasks.config as cfg
import numpy as np
from Tools.NetworkTypes import *
import gym

grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook


class PolicyNetwork(nn.Module):
    def __init__(self, env):
        super(PolicyNetwork, self).__init__()
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.n
        assert isinstance(env.action_space, gym.spaces.Discrete)

        hidden_size = cfg.gym.hidden_size
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x


    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob


class GymNetwork(nn.Module):
    def __init__(self, agent):
        super(GymNetwork, self).__init__()

        self.networkType = NetworkTypes.Torch

        self.agent = agent
        numS = 8
        bias = False
        self.wl1 = nn.Linear(
            agent.inputSize, out_features=numS, bias=bias).cuda()

        self.wl2 = nn.Linear(
            in_features=numS, out_features=numS, bias=bias)

        self.wly = nn.Linear(
            in_features=self.wl2.out_features, out_features=self.agent.outputSize, bias=bias).cuda()

    def forward(self, envInput, selfInput):
        l0 = torch.tanh(self.wl1(torch.cat((envInput, selfInput), dim=1)))
        l1 = torch.tanh(self.wl2(l0))
        ly = torch.tanh(self.wly(l1))
        output = F.softmax(ly, dim=1)
        return output, output
