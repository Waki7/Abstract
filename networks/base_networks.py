import logging
from typing import List

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
        # floating point precision, so need to set epislon
        self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=lr, eps=1.e-4)
        self.to(settings.DEVICE)
        self.half()  # convert to half precision
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

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
        encoding = F.relu(self.linear1(encoding))
        if self.use_lstm:
            if self.hidden_state is None:
                batch_size = encoding.shape[0]
                self.hidden_state = torch.zeros((batch_size, self.model_size)).to(**settings.ARGS)
                self.context = torch.zeros((batch_size, self.model_size)).to(**settings.ARGS)
            self.hidden_state, self.context = self.lstm.forward(x=encoding, hidden=self.hidden_state,
                                                                context=self.context)
            encoding = self.hidden_state
        encoding = self.linear2(encoding)
        return encoding


@register_network
class ActorFCNetwork(BaseNetwork):
    def __init__(self, in_shapes, out_shapes, cfg, **kwargs):
        super().__init__(in_shapes=in_shapes, out_shapes=out_shapes, cfg=cfg)
        self.n_actions = self.out_features
        self.create_optimizer()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        encoding = self.encode(features)
        features = F.softmax(encoding, dim=-1)
        return features


@register_network
class CriticFCNetwork(BaseNetwork):
    def __init__(self, in_shapes, out_shapes, cfg, **kwargs):
        super().__init__(in_shapes=in_shapes, out_shapes=out_shapes, cfg=cfg)
        self.create_optimizer()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        encoding = self.encode(features)
        return encoding
