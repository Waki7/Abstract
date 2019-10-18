import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.NetworkTypes import *
import gym_life.envs.life_channels as ch
import numpy as np

grads = {}
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
DTYPE = torch.float
ARGS = {'device': DEVICE, 'dtype': DTYPE}


def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook


class LifeNetwork(nn.Module):
    def __init__(self):
        super(LifeNetwork, self).__init__()

        self.env_in_channels = ch.AGENT_STATE_CHANNELS
        self.hidden_in_channels = [ch.See, ch.Hear, ch.Speak, ch.Feel]
        self.in_channels = np.concatenate((self.env_in_channels, self.hidden_in_channels))
        self.in_shapes = [len(input) for input in self.in_channels]
        self.hidden_in_channel_index = len(self.env_in_channels)

        self.action_channels = ch.AGENT_ACTION_CHANNELS
        self.hidden_out_channels = [ch.See, ch.Hear, ch.Speak, ch.Feel]
        self.out_channels = np.concatenate((self.action_channels, self.hidden_out_channels))
        self.out_shapes = [len(output) for output in self.out_channels]
        self.hidden_out_channel_index = len(self.action_channels)
        self.prediction_indexes = [(0, 5), (5, 6)]
        assert (self.prediction_indexes[-1][-1] == len(self.out_channels))

        self.in_vector_idx, self.out_vector_idx = [0], [0]
        self.in_vector_idx.extend(np.cumsum(self.in_shapes))
        self.out_vector_idx.extend(np.cumsum(self.out_shapes))

        self.env_in_size = self.in_vector_idx[len(self.env_in_channels)]
        self.hidden_in_size = np.sum(self.in_shapes[len(self.env_in_channels):])

        self.env_out_size = self.out_vector_idx[len(self.action_channels)]
        self.hidden_out_size = np.sum(self.out_shapes[len(self.action_channels):])

        self.in_size = self.in_vector_idx[-1]
        self.out_size = self.out_vector_idx[-1]

        self.networkType = NetworkTypes.Torch

        numS = 16
        bias = False
        l1_out_features = 0
        self.wl1 = []
        for channel_idx in range(0, len(self.in_channels)):
            self.wl1.append(nn.Linear(
                in_features=self.in_shapes[channel_idx], out_features=numS, bias=bias).cuda())
            l1_out_features += self.wl1[channel_idx].out_features

        self.wl2 = nn.Linear(
            in_features=l1_out_features, out_features=numS, bias=bias)

        self.wly = []
        for channel_idx in range(0, len(self.out_channels)):
            self.wly.append(nn.Linear(
                in_features=self.wl2.out_features, out_features=self.out_shapes[channel_idx], bias=bias).cuda())

    def forward(self, env_input, hidden_input):
        assert isinstance(hidden_input, torch.Tensor)
        env_input = torch.from_numpy(env_input, **ARGS).float().unsqueeze(0)

        l0, l1 = [], []
        input = torch.cat((env_input, hidden_input), dim=1)
        for channel_idx in range(0, len(self.in_channels)):
            neuron = self.wl1[channel_idx]
            l0.append(input[:, self.in_vector_idx[channel_idx]: self.in_vector_idx[
                channel_idx + 1]])  # todo fuck this might not be right fuck
            l1.append(torch.tanh(neuron(l0[channel_idx])))

        l1_cmbn = torch.cat(l1, dim=1)
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
        hidden = output[:, hiddenStartIndex:] #todo try to experiment with having all of the output part of hidden state.
        output = output[:, :hiddenStartIndex]
        return output, hidden

    def get_action_vector(self, output):
        preds = torch.argmax(output, dim=-1)
        Ytarg = np.zeros((1, self.out_size))
        if isinstance(preds, (tuple)):
            for pred in preds:
                Ytarg[0, pred] = 1.0  # promote the action being currently explored
        else:
            Ytarg[0, preds] = 1.0
        return Ytarg
