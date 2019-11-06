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

    def create_optimizer(self):
        pass
        # self.optim = getattr(torch.optim, self.cfg['optim'])(
        #     list(critic.parameters()), lr=self.cfg['lr'])


@register_network
class ActorFCNetwork(BaseNetwork):
    def __init__(self, n_features, n_actions, cfg):
        super().__init__(cfg)
        self.linear1 = nn.Linear(n_features, self.model_size)
        self.linear2 = nn.Linear(hidden_size, n_actions)

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
        hidden_size = cfg.gym.hidden_size
        self.linear1 = nn.Linear(n_features, self.model_size)
        self.linear2 = nn.Linear(hidden_size, critic_estimates)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim=1)
        return x


@register_network
class ACNetwork:  # actor critic method, parameterized baseline estimate with network
    def __init__(self, n_features, n_actions, critic_estimates, actor_cfg, critic_cfg):
        self.actor = ActorFCNetwork(n_features, n_actions, actor_cfg)
        self.critic = CriticFCNetwork(n_features, critic_estimates, critic_cfg)
        self.actor = None
        self.critic = None
