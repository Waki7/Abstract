import torch
import torch.nn as nn
import torch.nn.functional as F
import gym_life.envs.life_channels as ch
import numpy as np
import settings

class LifeNetwork(nn.Module):
    def __init__(self):
        super(LifeNetwork, self).__init__()
        # this gets broken down into sub categories below each for life
        default_fallback = ['env in channels', 'hidden in channels', ]
        # trhis gets broken down into sub categories below each for life
        default_input = ['action', 'focus', ' maybe feel> ']



        self.env_in_channels = ch.AGENT_STATE_CHANNELS
        self.hidden_in_channels = [ch.See, ch.Hear, ch.Speak, ch.Feel]
        self.in_channels = np.concatenate((self.env_in_channels, self.hidden_in_channels))
        self.in_shapes = [len(input) for input in self.in_channels]

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
        env_input = torch.from_numpy(env_input, **settings.ARGS).float().unsqueeze(0)

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

        # self.aux_reward = 0
        # if self.is_life_env:
        #     if self.model.get_env_pred_val() is not None:  # todo
        #         self.aux_reward += cfg.life.EXPLOITATION_PENALTY
        #     if cfg.self_reward_update:
        #         self.aux_reward += cfg.reward_prediction_discount * cfg.adjacent_reward_list[self.pred_feel_val.value]
        # return self.aux_reward

class life_rnn():
    def __init__(self):
        self.initial_state = torch.zeros((1, self.model.hidden_in_size), **settings.ARGS)
        self.hidden_states = [(None, self.initial_state)]
        optim = getattr(torch.optim, cfg.life.OPTIMIZER)(self.model.parameters(), lr=cfg.life.LR)


    def back_propagate(self):
        incremental_reward = 0
        while self.rewards:  # back prop through previous time steps
            discounted_reward = self.discount_factor * incremental_reward
            curr_reward = self.rewards.pop() if self.rewards else 0
            output = self.outputs.pop()
            hidden_states = self.hidden_states.pop()
            action = self.actions.pop()

            incremental_reward = curr_reward + discounted_reward
            loss = self.criterion(input=output, target=action)  # (f(s_t) - a_t)
            loss *= incremental_reward
            loss.backward(retain_graph=True)
            if self.backprop_through_input:
                if hidden_states[0] is None:
                    break
                if hidden_states[
                    0].grad is not None:  # can't clear gradient if it hasn't been back propagated once through
                    hidden_states[0].grad.data.zero_()
                curr_grad = hidden_states[0].grad
                hidden_states[1].backward(curr_grad, retain_graph=True)
        assert self.rewards == []

    def forward(self, env_input):
        hidden_input = self.hidden_states[-1][1].detach()
        hidden_input.requires_grad = True
        output, hidden_output = self.model.forward(env_input, hidden_input)
        action = torch.argmax(output)  # WHERE THE FUCK DO WE STORE THIS WHOLE SPECIFIC TO THE ENVIRONMENT BULLSHIT
        # action = self.model.get_action_vector(output)
        self.outputs.append(output)
        self.hidden_states.append((hidden_input, hidden_output))
        return action

