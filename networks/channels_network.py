import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import settings
from networks.base_networks import BaseNetwork
from networks.factory import register_network


def attention():
    attention_heads = 5
    attention_inputs = torch.tensor(list(range(attention_heads)))
    inputs = [1, 2, 3, 4, 5]
    attention_layer = nn.Linear(2, 1)
    attention_output = []
    for attention_input in attention_inputs:
        attention_attributes = []
        for input in inputs:
            full_input = torch.cat(attention_input, input)
            attention_attributes.append(attention_layer(full_input))
        attention_output.append(inputs * F.softmax(attention_attributes, dim=-1))


def attention2():
    attention_heads = 5
    inputs = [1, 2, 3, 4, 5]
    attention_layer = nn.Linear(2, 5)
    attention_output = []
    for input in inputs:
        attention_out = F.softmax(attention_layer(input), dim=-1)
        attention_output.append(inputs * F.softmax(attention_out, dim=-1))


@register_network
class Attention(BaseNetwork):
    def __init__(self, n_features, out_shape, cfg={},
                 in_channels=None, in_shapes=None, out_channels=None, out_shapes=None, **kwargs):
        super(Attention, self).__init__(cfg)
        self.attenion_head = nn.Linear(n_features, out_shape)
        self.create_optimizer()

    def forward(self, inputs):
        attention_inputs = torch.tensor(list(range(inputs.shape[-1]))).to(settings.DEVICE).float()
        attention_inputs = attention_inputs / inputs.shape[-1]

        attention_attributes = []
        for input, attention_input in zip(inputs, attention_inputs):
            full_input = torch.stack([attention_input, input])
            attention_attributes.append(self.attenion_head(full_input))
        attention_attributes = torch.cat(attention_attributes)
        weighted_attention = inputs * F.softmax(attention_attributes, dim=-1)
        # print(inputs)
        # print()
        # print(weighted_attention)
        # print(exit(9))
        return weighted_attention


@register_network
class ChannelNetworkBasic(BaseNetwork):
    def __init__(self, n_features, out_shape, cfg,
                 in_channels=None, in_shapes=None, out_channels=None, out_shapes=None, **kwargs):
        super(ChannelNetworkBasic, self).__init__(cfg)
        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        reward_embedding_size = cfg.get('reward_embedding_size', 1)
        self.model_size = cfg.get('model_size', 32)
        self.bias = cfg.get('bias', True)
        self.out_shape = out_shape

        ##########################################################################################
        # define channels
        ##########################################################################################
        super().__init__(cfg)
        self.n_actions = out_shape
        self.hidden_size = out_shape
        self.linear1 = nn.Linear(n_features + self.hidden_size, self.model_size)
        self.linear2 = nn.Linear(self.model_size, out_shape)
        self.hidden_state = torch.zeros((1, self.hidden_size)).to(settings.DEVICE)
        self.create_optimizer()

    def forward(self, env_input, **kwargs):
        x = torch.cat([env_input, self.hidden_state], dim=-1)
        x = F.leaky_relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim=-1)
        self.hidden_state = x
        return x, None

    def prune(self):
        self.hidden_state = self.hidden_state.detach()

    def reset_state(self):
        self.hidden_state = torch.zeros((1, self.hidden_size)).to(settings.DEVICE)


@register_network
class ChannelNetworkSwitch(BaseNetwork):
    def __init__(self, n_features, out_shape, cfg,
                 in_channels=None, in_shapes=None, out_channels=None, out_shapes=None, **kwargs):
        super(ChannelNetworkSwitch, self).__init__(cfg)
        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.reward_embedding_size = cfg.get('reward_embedding_size', 1)
        self.model_size = cfg.get('model_size', 32)
        self.bias = cfg.get('bias', True)
        self.state_embedding_size = cfg.get('state_embedding_size', -1)
        hidden_features = n_features + out_shape
        if self.state_embedding_size > 0:
            hidden_features = self.state_embedding_size

        ##########################################################################################
        # define channels
        ##########################################################################################
        self.in_channels = ['env in channels', 'reward', 'hidden_state'] \
            if in_channels is None else in_channels
        self.n_in_channels = len(self.in_channels)
        self.in_shapes = [n_features, self.reward_embedding_size, hidden_features] \
            if in_shapes is None else in_shapes
        self.in_shape = n_features + self.reward_embedding_size
        self.hidden_in_shape = sum(
            [shape for shape, channel in zip(self.in_shapes, self.in_channels) if 'hidden' in channel])

        self.out_channels = ['action', 'focus/future'] \
            if out_channels is None else out_channels
        self.n_out_channels = len(self.out_channels)
        self.out_shapes = [out_shape, hidden_features] \
            if out_shapes is None else out_shapes
        self.n_actions = out_shape
        self.hidden_n_actions = sum(self.out_shapes[1:])

        self.in_vector_idx, self.out_vector_idx = [0], [0]
        self.in_vector_idx.extend(np.cumsum(self.in_shapes))
        self.out_vector_idx.extend(np.cumsum(self.out_shapes))
        self.in_size = self.in_vector_idx[-1]
        self.out_size = self.out_vector_idx[-1]

        ##########################################################################################
        # create layers
        ##########################################################################################
        l1_out_features = 0
        self.fc1s = nn.ModuleList()
        for channel_idx in range(0, self.n_in_channels):
            new_layer = nn.Linear(
                in_features=self.in_shapes[channel_idx], out_features=self.model_size, bias=self.bias)
            logging.info(
                'new channel for {}, with size {}'.format(self.in_channels[channel_idx], self.in_shapes[channel_idx]))
            self.fc1s.append(new_layer)
            l1_out_features += self.fc1s[channel_idx].out_features

        self.cmbn_fc2 = nn.Linear(
            in_features=l1_out_features, out_features=self.model_size, bias=self.bias)

        self.out_fcs = nn.ModuleList()
        for channel_idx in range(0, self.n_out_channels):
            new_layer = nn.Linear(
                in_features=self.cmbn_fc2.out_features, out_features=self.out_shapes[channel_idx], bias=self.bias)
            self.out_fcs.append(new_layer)

        self.hidden_state = torch.zeros((1, self.hidden_in_shape)).to(settings.DEVICE)  # 1 for batch size

        self.empty_hidden = torch.zeros((1, self.hidden_in_shape)).to(settings.DEVICE)

        self.create_optimizer()

    def forward(self, env_input, reward=None, hidden_input=None, action=None):
        if hidden_input is None:
            hidden_input = self.hidden_state
        if action is not None:
            action = action - self.n_actions  # offset by the number of env actions
            hidden_mask = self.empty_hidden.detach().clone()
            hidden_mask[:, action] = 1.0
            self.hidden_state = self.hidden_state * hidden_mask
        channel_inputs, layer1 = [], []

        input = torch.cat((env_input, hidden_input, reward), dim=-1)
        for channel_idx in range(0, len(self.in_channels)):
            fc1 = self.fc1s[channel_idx]
            channel_input = input[:, self.in_vector_idx[channel_idx]: self.in_vector_idx[channel_idx + 1]]
            layer1.append(fc1(channel_input))

        l1_cmbn = F.leaky_relu(torch.cat(layer1, dim=-1))
        l2 = F.leaky_relu(self.cmbn_fc2(l1_cmbn))
        ly = []

        for channel_idx in range(0, len(self.out_channels)):
            neuron = self.out_fcs[channel_idx]
            outNeuron = neuron(l2)
            ly.append(outNeuron)
            # ly[outputChannel] = F.softmax(outNeuron, dim=1)

        output = torch.cat(ly, dim=1)

        prob_output = F.softmax(output,
                                dim=-1)  # todo try to experiment with having all of the output part of hidden state.

        action = prob_output[:, :self.out_vector_idx[1]]
        hidden_concepts = prob_output[:, self.out_vector_idx[1]:self.out_vector_idx[2]]

        self.hidden_state = hidden_concepts

        return action, hidden_concepts

    def prune(self):
        self.hidden_state = self.hidden_state.detach()
        # pass

    def reset_state(self):
        self.hidden_state = torch.zeros((1, self.hidden_in_shape)).to(settings.DEVICE)
    # def update_paramters(self):
    #     pass


@register_network
class ChannelNetwork(BaseNetwork):
    def __init__(self, n_features, out_shape, cfg,
                 in_channels=None, in_shapes=None, out_channels=None, out_shapes=None, **kwargs):
        super(ChannelNetwork, self).__init__(cfg)
        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.reward_embedding_size = cfg.get('reward_embedding_size', 1)
        self.model_size = cfg.get('model_size', 32)
        self.bias = cfg.get('bias', True)

        ##########################################################################################
        # define channels
        ##########################################################################################
        self.in_channels = ['env in channels', 'hidden in channels', 'hidden reward'] \
            if in_channels is None else in_channels
        self.n_in_channels = len(self.in_channels)
        self.in_shapes = [n_features, out_shape, self.reward_embedding_size] \
            if in_shapes is None else in_shapes

        self.out_channels = ['action', 'focus', 'future', 'maybefeel'] \
            if out_channels is None else out_channels
        self.n_out_channels = len(self.out_channels)
        self.out_shapes = [out_shape, out_shape, n_features, self.reward_embedding_size] \
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
            logging.info(
                'new channel for {}, with size {}'.format(self.in_channels[channel_idx], self.in_shapes[channel_idx]))
            self.fc1s.append(new_layer)
            l1_out_features += self.fc1s[channel_idx].out_features

        self.cmbn_fc2 = nn.Linear(
            in_features=l1_out_features, out_features=self.model_size, bias=self.bias)

        self.out_fcs = nn.ModuleList()
        for channel_idx in range(0, self.n_out_channels):
            new_layer = nn.Linear(
                in_features=self.cmbn_fc2.out_features, out_features=self.out_shapes[channel_idx], bias=self.bias)
            self.out_fcs.append(new_layer)

        self.hidden_state = torch.zeros((1, self.hidden_in_shape)).to(settings.DEVICE)  # 1 for batch size

        self.create_optimizer()

    def forward(self, env_input, hidden_input=None):
        if hidden_input is None:
            hidden_input = self.hidden_state
        channel_inputs, layer1 = [], []
        # input = torch.cat((env_input, hidden_input), dim=-1)
        input = torch.cat((env_input, hidden_input), dim=-1)
        for channel_idx in range(0, len(self.in_channels)):
            fc1 = self.fc1s[channel_idx]
            channel_input = input[:, self.in_vector_idx[channel_idx]: self.in_vector_idx[channel_idx + 1]]
            layer1.append(fc1(channel_input))

        l1_cmbn = F.leaky_relu(torch.cat(layer1, dim=-1))
        l2 = F.leaky_relu(self.cmbn_fc2(l1_cmbn))
        ly = []

        for channel_idx in range(0, len(self.out_channels)):
            neuron = self.out_fcs[channel_idx]
            outNeuron = neuron(l2)
            ly.append(outNeuron)
            # ly[outputChannel] = F.softmax(outNeuron, dim=1)
        hidden_state_idx = self.n_actions

        output = torch.cat(ly, dim=1)

        # output = F.softmax(output, dim=1)
        # hidden = output[:,
        #          hiddenStartIndex:]  # todo try to experiment with having all of the output part of hidden state.
        # output = output[:, :hiddenStartIndex]
        #
        hidden_think = output[:, self.out_vector_idx[1]:self.out_vector_idx[2]]
        hidden_pred = output[:, self.out_vector_idx[2]:self.out_vector_idx[3]]
        hidden_reward = output[:, self.out_vector_idx[3]:]

        # hidden = F.softmax(output[:,
        #                    hidden_state_idx - self.reward_embedding_size:],
        #                    dim=-1)  # todo try to experiment with having all of the output part of hidden state.

        action = F.softmax(output[:, :hidden_state_idx], dim=-1)

        self.hidden_state = output[:, hidden_state_idx:]

        return action, hidden_think, hidden_pred, hidden_reward

    def prune(self):
        self.hidden_state = self.hidden_state.detach()
        # pass

    def reset_state(self):
        self.hidden_state = torch.zeros((1, self.hidden_in_shape)).to(settings.DEVICE)
    # def update_paramters(self):
    #     pass


@register_network
class CriticChannels(BaseNetwork):
    def __init__(self, out_shape, cfg, n_features, **kwargs):
        super().__init__(cfg)

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.state_embedding_size = cfg.get('state_embedding_size', -1)
        hidden_features = n_features
        if self.state_embedding_size > 0:
            hidden_features = self.state_embedding_size

        ##########################################################################################
        # create layers
        ##########################################################################################
        self.linear_pre_prob = nn.Linear(n_features, hidden_features)
        # soft max will be applied to the above output
        self.linear1 = nn.Linear(hidden_features, self.model_size)
        self.linear2 = nn.Linear(self.model_size, out_shape)
        self.create_optimizer()

    def forward(self, x):
        x = F.leaky_relu(self.linear_pre_prob(x))
        probs = F.softmax(x, dim=-1)
        x = F.leaky_relu(self.linear1(probs))
        x = F.leaky_relu(self.linear2(x))
        return x, probs

    def forward_prob(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        return x
