from networks.factory import register_network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import settings
import gym


class BaseNetwork(nn.Module):
    def __init__(self, cfg):
        super(BaseNetwork, self).__init__()
        self.cfg = cfg
        self.model_size = cfg.get('model_size', settings.defaults.MODEL_SIZE)
        self.optimizer = None  # call create_optimizer at end of your implementation's init

    def create_optimizer(self):
        lr = self.cfg.get('lr', settings.defaults.LR)
        optimizer = self.cfg.get('optimizer', settings.defaults.OPTIMIZER)
        self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=lr)
        self.to(settings.DEVICE)

    def update_parameters(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


@register_network
class ActorFCNetwork(BaseNetwork):
    def __init__(self, n_features, n_actions, cfg):
        super().__init__(cfg)
        self.linear1 = nn.Linear(n_features, self.model_size)
        self.linear2 = nn.Linear(self.model_size, n_actions)
        self.create_optimizer()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim=1)
        return x

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        prob = probs.squeeze(0)[highest_prob_action]
        return highest_prob_action, prob


@register_network
class CriticFCNetwork(BaseNetwork):
    def __init__(self, n_features, critic_estimates, cfg):
        super().__init__(cfg)
        self.linear1 = nn.Linear(n_features, self.model_size)
        self.linear2 = nn.Linear(self.model_size, critic_estimates)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim=1)
        return x


@register_network
class ACNetwork(BaseNetwork):  # actor critic method, parameterized baseline estimate with network
    def __init__(self, n_features, n_actions, critic_estimates, cfg):
        super().__init__(cfg)
        self.n_actions = n_actions
        self.linear_shared = nn.Linear(n_features, self.model_size)
        self.linear_actor = nn.Linear(self.model_size, n_actions)
        self.linear_critic = nn.Linear(self.model_size, critic_estimates)
        super().create_optimizer()

    def forward(self, x):
        x = self.linear_shared(x)
        actor_estimate = F.softmax(self.linear_actor(x), dim=-1)
        critic_estimate = self.linear_critic(x)
        return actor_estimate, critic_estimate
