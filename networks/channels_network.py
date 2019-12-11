import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import settings
from networks.base_networks import BaseNetwork
from networks.factory import register_network


@register_network
class ChannelNetwork(BaseNetwork):
    def __init__(self, n_features, out_shape, cfg,
                 in_channels=None, in_shapes=None, out_channels=None, out_shapes=None, **kwargs):
        super(ChannelNetwork, self).__init__(cfg)
        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        reward_embedding_size = cfg.get('reward_embedding_size', 1)
        self.model_size = cfg.get('model_size', 32)
        self.bias = cfg.get('bias', True)

        ##########################################################################################
        # define channels
        ##########################################################################################
        self.in_channels = ['env in channels', 'hidden in channels', ] \
            if in_channels is None else in_channels
        self.n_in_channels = len(self.in_channels)
        self.in_shapes = [n_features, out_shape, reward_embedding_size] \
            if in_shapes is None else in_shapes

        self.out_channels = ['action', 'focus', ' maybefeel'] \
            if out_channels is None else out_channels
        self.n_out_channels = len(self.out_channels)
        self.out_shapes = [out_shape, out_shape + n_features, reward_embedding_size] \
            if out_shapes is None else out_shapes

        self.in_vector_idx, self.out_vector_idx = [0], [0]
        self.in_vector_idx.extend(np.cumsum(self.in_shapes))
        self.out_vector_idx.extend(np.cumsum(self.out_shapes))
        self.in_size = self.in_vector_idx[-1]
        self.out_size = self.out_vector_idx[-1]

        self.in_shape = n_features
        self.hidden_in_shape = sum(self.in_shapes[1:])

        self.n_actions = out_shape
        self.hidden_n_actions = sum(self.out_shapes[1:])

        ##########################################################################################
        # create layers
        ##########################################################################################
        l1_out_features = 0
        self.fc1s = nn.ModuleList()
        for channel_idx in range(0, self.n_in_channels):
            new_layer = nn.Linear(
                in_features=self.in_shapes[channel_idx], out_features=self.model_size, bias=self.bias)
            self.fc1s.append(new_layer)
            l1_out_features += self.fc1s[channel_idx].out_features

        self.cmbn_fc2 = nn.Linear(
            in_features=l1_out_features, out_features=self.model_size, bias=self.bias)

        self.out_fcs = nn.ModuleList()
        for channel_idx in range(0, self.n_out_channels):
            new_layer = nn.Linear(
                in_features=self.cmbn_fc2.out_features, out_features=self.out_shapes[channel_idx], bias=self.bias)
            self.out_fcs.append(new_layer)

        self.hidden_input = torch.zeros((1, self.hidden_in_shape)).to(settings.DEVICE)  # 1 for batch size

        self.create_optimizer()

    def forward(self, env_input, hidden_input=None):
        if hidden_input is None:
            hidden_input = self.hidden_input
        channel_inputs, layer1 = [], []
        input = torch.cat((env_input, hidden_input), dim=-1)
        for channel_idx in range(0, len(self.in_channels)):
            fc1 = self.fc1s[channel_idx]
            channel_input = input[:, self.in_vector_idx[channel_idx]: self.in_vector_idx[channel_idx + 1]]
            layer1.append(torch.tanh(fc1(channel_input)))

        l1_cmbn = torch.cat(layer1, dim=-1)
        l2 = torch.tanh(self.cmbn_fc2(l1_cmbn))
        ly = []

        for channel_idx in range(0, len(self.out_channels)):
            neuron = self.out_fcs[channel_idx]
            outNeuron = neuron(l2)
            ly.append(outNeuron)
            # ly[outputChannel] = F.softmax(outNeuron, dim=1)

        ly_cmbn = torch.cat(ly, dim=1)
        output = F.softmax(ly_cmbn, dim=1)
        hiddenStartIndex = self.n_actions
        hidden = output[:,
                 hiddenStartIndex:]  # todo try to experiment with having all of the output part of hidden state.
        output = output[:, :hiddenStartIndex]
        return output, hidden

    def prune(self):
        self.hidden_input = self.hidden_input.detach()

    # def update_paramters(self):
    #     pass
