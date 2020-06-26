import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import networks.network_blocks as nets
import settings
import utils.model_utils as model_utils
from networks.factory import register_network


class BaseNetwork(nn.Module):
    def __init__(self, in_shapes, out_shapes, cfg={}):
        super(BaseNetwork, self).__init__()
        self.cfg = cfg
        self.extra_parameters = nn.ParameterList()
        self.in_shapes = in_shapes
        self.in_features = model_utils.sum_multi_modal_shapes(in_shapes)
        self.out_features = model_utils.sum_multi_modal_shapes(out_shapes)

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.model_size = cfg.get('model_size', settings.defaults.MODEL_SIZE)
        self.gradient_clip = cfg.get('gradient_clip', settings.defaults.GRADIENT_CLIP)
        self.use_lstm = cfg.get('use_lstm', False)
        logging.debug(' model_size : {}'.format(self.model_size))
        logging.debug(' gradient_clip : {}'.format(self.gradient_clip))
        logging.debug(' use_lstm : {}'.format(self.use_lstm))

        ##########################################################################################
        # basic encoding layers
        ##########################################################################################
        self.linear1 = nn.Linear(self.in_features, self.model_size)
        if self.use_lstm:
            self.lstm = nets.LSTM(in_features=self.model_size, hidden_features=self.model_size)
            self.hidden_state = None
            self.context = None
        self.linear2 = nn.Linear(self.model_size, self.out_features)

        self.optimizer = None  # call create_optimizer at end of your implementation's init

    def create_optimizer(self):
        lr = self.cfg.get('lr', settings.defaults.LR)
        optimizer = self.cfg.get('optimizer', settings.defaults.OPTIMIZER)
        self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=lr)
        self.to(settings.DEVICE)

    def add_parameters(self, parameters):
        self.extra_parameters.extend(parameters)
        self.create_optimizer()  # recreate optimizer due to neew parameters

    def update_parameters(self):
        torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def reset_time(self):
        self.hidden_state = None
        self.context = None

    def encode(self, features: List[torch.Tensor]) -> torch.Tensor:
        encoding = torch.cat([x.flatten(start_dim=1) for x in features], dim=-1)
        if self.use_lstm:
            if self.hidden_state is None:
                batch_size = encoding.shape[0]
                self.hidden_state = torch.zeros((batch_size, self.model_size)).to(settings.DEVICE)
                self.context = torch.zeros((batch_size, self.model_size)).to(settings.DEVICE)
            encoding = self.lstm.forward(x=encoding, hidden=self.hidden_state, context=self.context)
        encoding = F.relu(self.linear1(encoding))
        return encoding


@register_network
class ActorFCNetwork(BaseNetwork):
    def __init__(self, in_shapes, out_shapes, cfg, **kwargs):
        super().__init__(in_shapes=in_shapes, out_shapes=out_shapes, cfg=cfg)
        self.n_actions = self.out_features
        self.create_optimizer()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        encoding = self.encode(features)
        features = F.softmax(self.linear2(encoding), dim=-1)
        return features

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        prob = probs.squeeze(0)[highest_prob_action]
        return highest_prob_action, prob


@register_network
class CriticFCNetwork(BaseNetwork):
    def __init__(self, in_shapes, out_shapes, cfg, **kwargs):
        super().__init__(in_shapes=in_shapes, out_shapes=out_shapes, cfg=cfg)
        self.create_optimizer()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        return self.encode(features)


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
