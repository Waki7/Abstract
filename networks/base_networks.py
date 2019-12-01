import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import settings
from networks.factory import register_network


class BaseNetwork(nn.Module):
    def __init__(self, cfg):
        super(BaseNetwork, self).__init__()
        self.cfg = cfg

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.model_size = cfg.get('model_size', settings.defaults.MODEL_SIZE)
        self.gradient_clip = cfg.get('gradient_clip', settings.defaults.GRADIENT_CLIP)
        logging.debug(' model_size : ', self.model_size, '\n')
        logging.debug(' gradient_clip : ', self.gradient_clip, '\n')

        self.optimizer = None  # call create_optimizer at end of your implementation's init

    def create_optimizer(self):
        lr = self.cfg.get('lr', settings.defaults.LR)
        optimizer = self.cfg.get('optimizer', settings.defaults.OPTIMIZER)
        self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=lr)
        self.to(settings.DEVICE)

    def update_parameters(self):
        torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()


@register_network
class ActorFCNetwork(BaseNetwork):
    def __init__(self, out_shape, cfg, n_features=0, **kwargs):
        super().__init__(cfg)
        self.n_actions = out_shape
        self.linear1 = nn.Linear(n_features, self.model_size)
        self.linear2 = nn.Linear(self.model_size, out_shape)
        self.create_optimizer()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim=-1)
        return x

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        prob = probs.squeeze(0)[highest_prob_action]
        return highest_prob_action, prob


@register_network
class CriticFCNetwork(BaseNetwork):
    def __init__(self, out_shape, cfg, n_features, **kwargs):
        super().__init__(cfg)
        self.linear1 = nn.Linear(n_features, self.model_size)
        self.linear2 = nn.Linear(self.model_size, out_shape)
        self.create_optimizer()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


@register_network
class CENetwork(BaseNetwork):
    def __init__(self, n_features, out_shape, out_shape2, cfg, **kwargs):
        super().__init__(cfg)
        critic_estimates = out_shape
        aux_estimates = out_shape2
        self.linear_shared = nn.Linear(n_features, self.model_size)
        self.linear_critic = nn.Linear(self.model_size, critic_estimates)
        self.linear_estimator = nn.Linear(self.model_size, aux_estimates)
        self.create_optimizer()

    def forward(self, x):
        x = F.relu(self.linear_shared(x))
        critic_estimate = self.linear_critic(x)
        linear_estimate = self.linear_estimator(x)
        return critic_estimate, linear_estimate


@register_network
class ACENetwork(BaseNetwork):
    def __init__(self, n_features, out_shape, out_shape2, out_shape3, cfg, **kwargs):
        super().__init__(cfg)
        n_actions = out_shape
        critic_estimates = out_shape2
        aux_estimates = out_shape3
        self.n_actions = n_actions

        split_size = self.model_size // 1

        self.shared_1 = nn.Linear(n_features, split_size)
        self.actor_1 = nn.Linear(n_features, split_size)
        self.critic_1 = nn.Linear(n_features, split_size)
        self.aux_1 = nn.Linear(n_features, split_size)

        self.actor_out = nn.Linear(self.model_size, n_actions)
        self.critic_out = nn.Linear(self.model_size, critic_estimates)
        self.aux_out = nn.Linear(self.model_size, aux_estimates)
        self.create_optimizer()

    def forward(self, x):
        # x_shared = F.relu(self.shared_1(x))
        x_actor = F.relu(self.actor_1(x))
        x_critic = F.relu(self.critic_1(x))
        x_aux = F.relu(self.aux_1(x))

        # x_actor = torch.cat([x_shared, x_actor], dim=-1)
        # x_critic = torch.cat([x_shared, x_critic], dim=-1)
        # x_aux = torch.cat([x_shared, x_aux], dim=-1)

        actor_estimate = F.softmax(self.actor_out(x_actor), dim=-1)
        critic_estimate = self.critic_out(x_critic)
        aux_estimates = self.aux_out(x_aux)
        return actor_estimate, critic_estimate, aux_estimates


@register_network
class CRANetwork(BaseNetwork):
    def __init__(self, n_features, out_shape, out_shape2, out_shape3, cfg, **kwargs):
        super().__init__(cfg)
        self.n_actions = out_shape
        self.n_critic_estimates = out_shape2
        self.n_aux_estimates = out_shape3
        self.linear_shared = nn.Linear(n_features + self.n_actions + self.n_critic_estimates + self.n_aux_estimates,
                                       self.model_size)
        self.linear_actor = nn.Linear(self.model_size, self.n_actions)
        self.linear_critic = nn.Linear(self.model_size, self.n_critic_estimates)
        self.linear_estimator = nn.Linear(self.model_size, self.n_aux_estimates)

        self.actor_estimate = torch.zeros(self.n_actions).to(settings.DEVICE).unsqueeze(0)
        self.critic_estimate = torch.zeros(self.n_critic_estimates).to(settings.DEVICE).unsqueeze(0)
        self.aux_estimates = torch.zeros(self.n_aux_estimates).to(settings.DEVICE).unsqueeze(0)

        self.create_optimizer()

    def forward(self, x):
        x = torch.cat([x, self.actor_estimate, self.critic_estimate, self.aux_estimates], dim=-1)
        x = F.relu(self.linear_shared(x))
        self.actor_estimate = F.softmax(self.linear_actor(x), dim=-1)
        self.critic_estimate = self.linear_critic(x)
        self.aux_estimates = self.linear_estimator(x)
        return self.actor_estimate, self.critic_estimate, self.aux_estimates

    def update_parameters(self):
        super().update_parameters()
        self.actor_estimate = torch.zeros(self.n_actions).to(settings.DEVICE).unsqueeze(0)
        self.critic_estimate = torch.zeros(self.n_critic_estimates).to(settings.DEVICE).unsqueeze(0)
        self.aux_estimates = torch.zeros(self.n_aux_estimates).to(settings.DEVICE).unsqueeze(0)


@register_network
class ACNetwork(BaseNetwork):  # actor critic method, parameterized baseline estimate with network
    def __init__(self, n_features, out_shape, out_shape2, cfg, **kwargs):
        super().__init__(cfg)
        n_actions = out_shape
        critic_estimates = out_shape2
        self.n_actions = n_actions
        self.linear_shared = nn.Linear(n_features, self.model_size)
        self.linear_actor = nn.Linear(self.model_size, n_actions)
        self.linear_critic = nn.Linear(self.model_size, critic_estimates)
        self.create_optimizer()

    def forward(self, x):
        x = F.relu(self.linear_shared(x))
        actor_estimate = F.softmax(self.linear_actor(x), dim=-1)
        critic_estimate = self.linear_critic(x)
        return actor_estimate, critic_estimate
