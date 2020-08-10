import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import networks.network_blocks as nets
import settings
from networks.net_factory import register_network
from networks.network_interface import NetworkInterface


class BaseFCNetwork(NetworkInterface):
    def __init__(self, in_shapes, out_shapes, cfg={}):
        super(BaseFCNetwork, self).__init__(in_shapes=in_shapes, out_shapes=out_shapes, cfg=cfg)

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.model_size = cfg.get('model_size', settings.defaults.MODEL_SIZE)
        self.use_lstm = cfg.get('use_lstm', False)
        logging.debug(' model_size : {}'.format(self.model_size))
        logging.debug(' gradient_clip : {}'.format(self.gradient_clip))
        logging.debug(' use_lstm : {}'.format(self.use_lstm))

        ##########################################################################################
        # basic encoding layers
        ##########################################################################################
        self.linear1 = nn.Linear(self.in_features, self.model_size)
        if self.use_lstm:
            self.lstm: nets.LSTM = nets.LSTM(in_features=self.model_size, hidden_features=self.model_size)
            self.hidden_state = None
            self.context = None
        self.linear2 = nn.Linear(self.model_size, self.out_features)

        self.optimizer = None  # call create_optimizer at end of your implementation's init

    def add_parameters(self, parameters):
        self.extra_parameters.extend(parameters)
        self.create_optimizer()  # recreate optimizer due to neew parameters

    def reset_time(self):
        self.hidden_state = None
        self.context = None

    def encode(self, features: List[torch.Tensor]) -> torch.Tensor:
        encoding = torch.cat([x.flatten(start_dim=1) for x in features], dim=-1)
        encoding = F.relu(self.linear1(encoding))
        if self.use_lstm:
            if self.hidden_state is None:
                batch_size = encoding.shape[0]
                self.hidden_state, self.context = self.lstm.get_zero_input(batch_size)
            self.hidden_state, self.context = self.lstm.forward(x=encoding, hidden=self.hidden_state,
                                                                context=self.context)
            encoding = self.hidden_state
        encoding = self.linear2(encoding)
        return encoding


@register_network
class ActorFCNetwork(BaseFCNetwork):
    def __init__(self, in_shapes, out_shapes, cfg, **kwargs):
        super().__init__(in_shapes=in_shapes, out_shapes=out_shapes, cfg=cfg)
        self.n_actions = self.out_features
        self.create_optimizer()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        encoding = self.encode(features)
        features = F.softmax(encoding, dim=-1)
        return features


@register_network
class CriticFCNetwork(BaseFCNetwork):
    def __init__(self, in_shapes, out_shapes, cfg, **kwargs):
        super().__init__(in_shapes=in_shapes, out_shapes=out_shapes, cfg=cfg)
        self.create_optimizer()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        encoding = self.encode(features)
        return encoding
