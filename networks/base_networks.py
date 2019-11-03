from networks.factory import register_network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import config as cfg
import numpy as np
import gym

grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook


class BaseNetwork(nn.Module):
    def __init__(self, cfg):
        super(BaseNetwork, self).__init__()
        self.cfg = cfg
        self.model_size = cfg['model_size']

    def create_optimizer(self):
        pass
        # self.optim = getattr(torch.optim, self.cfg['optim'])(
        #     list(critic.parameters()), lr=self.cfg['lr'])


@register_network
class ActorFCNetwork(BaseNetwork):
    def __init__(self, n_features, n_actions, cfg):
        super(ActorFCNetwork, self).__init__(cfg)
        hidden_size = cfg.gym.hidden_size
        self.num_actions = num_actions
        self.linear1 = nn.Linear(n_features, self.hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.train()

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        prob = probs.squeeze(0)[highest_prob_action]
        return highest_prob_action, prob

class CriticFCNetwork():
    def __init__(self, n_features, n_estimates, cfg):
        super(ActorFCNetwork, self).__init__(cfg)
        hidden_size = cfg.gym.hidden_size
        self.num_actions = num_actions
        self.linear1 = nn.Linear(n_features, self.hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.train()


@register_network
class ACNetwork(BaseNetwork):  # actor critic method, parameterized baseline estimate with network
    def __init__(self, n_features, n_actions, critic_estimates, cfg):
        super(ACNetwork, self).__init__(cfg)
        self.shared_dense = nn.Linear(n_features, self.hi)
        self.critic_dense = nn.Linear(self.model_size)

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
