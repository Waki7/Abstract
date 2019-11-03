import torch
import torch.nn as nn
import torch.nn.functional as F
import gym_life.envs.life_channels as ch
import numpy as np
import settings


class ChannelNetwork(nn.Module):
    def __init__(self, in_shape, out_shape, cfg,
                 in_channels=None, in_shapes=None, out_channels=None, out_shapes=None):
        super(ChannelNetwork, self).__init__()
        reward_embedding_size = cfg.get('reward_embedding_size', 1)

        self.in_channels = ['env in channels', 'hidden in channels', ] \
            if in_channels is None else in_channels
        self.in_shapes = [in_shape, out_shape, reward_embedding_size] \
            if in_shapes is None else in_shapes

        self.out_channels = ['action', 'focus', ' maybe feel'] \
            if out_channels is None else out_channels
        self.out_shapes = [out_shape, out_shape + in_shape, reward_embedding_size] \
            if out_shapes is None else out_shapes

        self.in_vector_idx, self.out_vector_idx = [0], [0]
        self.in_vector_idx.extend(np.cumsum(self.in_shapes))
        self.out_vector_idx.extend(np.cumsum(self.out_shapes))
        self.in_size = self.in_vector_idx[-1]
        self.out_size = self.out_vector_idx[-1]

        self.env_in_size = in_shape
        self.hidden_in_size = sum(self.in_shapes[1:])

        self.env_out_size = out_shape
        self.hidden_out_size = sum(self.out_shapes[1:])

        self.model_size = cfg.get('model_size', 32)
        self.bias = cfg('bias', True)

        l1_out_features = 0
        self.wl1 = []
        for channel_idx in range(0, len(self.in_channels)):
            self.wl1.append(nn.Linear(
                in_features=self.in_shapes[channel_idx], out_features=self.model_size, bias=self.bias).cuda())
            l1_out_features += self.wl1[channel_idx].out_features

        self.wl2 = nn.Linear(
            in_features=l1_out_features, out_features=self.model_size, bias=self.bias)

        self.wly = []
        for channel_idx in range(0, len(self.out_channels)):
            self.wly.append(nn.Linear(
                in_features=self.wl2.out_features, out_features=self.out_shapes[channel_idx], bias=self.bias).cuda())

    def get_predictions(self, env_input):
        env_input = torch.from_numpy(env_input, **settings.ARGS).float().unsqueeze(0)

        channel_inputs, layer1 = [], []
        input = torch.cat((env_input, self.hidden_input), dim=1)
        for channel_idx in range(0, len(self.in_channels)):
            neuron = self.wl1[channel_idx]
            channel_inputs.append(input[:, self.in_vector_idx[channel_idx]: self.in_vector_idx[channel_idx + 1]])
            layer1.append(torch.tanh(neuron([channel_idx])))

        l1_cmbn = torch.cat(layer1, dim=-1)
        l2 = torch.tanh(self.wl2(l1_cmbn))
        ly = []

        for channel_idx in range(0, len(self.out_channels)):
            neuron = self.wly[channel_idx]
            outNeuron = neuron(l2)
            ly.append(outNeuron)
            # ly[outputChannel] = F.softmax(outNeuron, dim=1)

        ly_cmbn = torch.cat(ly, dim=1)
        output = F.softmax(ly_cmbn, dim=1)
        hiddenStartIndex = self.out_vector_idx[self.hidden_out_channel_index]
        hidden = output[:,
                 hiddenStartIndex:]  # todo try to experiment with having all of the output part of hidden state.
        output = output[:, :hiddenStartIndex]
        return output, hidden

    def update_paramters(self):
        pass
