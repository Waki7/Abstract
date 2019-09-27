import numpy as np
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import gym_life.envs.life_env as life_env
import gym
from Tools.TimeBuffer import TimeBuffer
import RLTasks.config as cfg
from tensorboardX import SummaryWriter

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
type = torch.float
args = {'device': device, 'dtype': type}


def logLoss(output, target):
    loss = torch.sum(torch.log(output))
    return loss


class CRAgent():
    # this agent can work with environments x, y, z (life and gym envs)
    # todo move cragent controller here, and move this stuff in life network
    # try to make the encoding part separate
    def __init__(self, model, env):
        self.model = model
        self.env = env

        self.pred_val, self.pred_feel_val = None, None
        self.reward = 0

        self.initial_state = torch.zeros((1, self.model.hidden_in_size), **args)
        self.hidden_states = [(None, self.initial_state)]
        self.outputs = []
        self.rewards = []
        self.actions = []

        self.testing_rewards = TimeBuffer(cfg.rewards_eval_window)
        self.mean_testing_rewards = []
        self.t = 0

        getattr(torch.optim, cfg.life.OPTIMIZER)(self.model.parameters(), lr=cfg.life.LR)
        self.criterion = nn.MSELoss()

        self.is_life_env = isinstance(self.env, life_env.LifeEnv)
        if not self.is_life_env:
            assert isinstance(self.env.action_space, gym.spaces.Discrete)
            self.num_actions = env.action_space.n

        self.writer = SummaryWriter()

    def step(self, env_input):
        action = self.forward(env_input)
        self.actions.append(action)
        self.t += 1
        return action

    def update_policy(self, env_reward, episode_end = True):
        self.env_reward = env_reward
        if self.t > 0:
            env_reward, aux_reward = self.calc_aux_reward()
            reward = env_reward + aux_reward
            self.store_results(env_reward)
            self.rewards.append(reward)
            if self.should_update_weights(reward, episode_end):
                self.back_propagate()
                self.model.opt.step()
                self.model.opt.zero_grad()
                self.t = 0

    def forward(self, env_input):
        hidden_input = self.hidden_states[-1][1].detach()
        hidden_input.requires_grad = True
        output, hidden_output = self.model.forward(env_input, hidden_input)
        action = self.model.get_action_vector(output)
        self.outputs.append(output)
        self.hidden_states.append((hidden_input, hidden_output))
        return action

    def back_propagate(self):
        incremental_reward = 0
        while self.rewards:  # back prop through previous time steps
            discounted_reward = cfg.discount_factor * incremental_reward
            curr_reward = self.rewards.pop() if self.rewards else 0
            output = self.outputs.pop()
            hidden_states = self.hidden_states.pop()
            action = self.actions.pop()

            incremental_reward = curr_reward + discounted_reward
            loss = self.criterion(input=output, target=action)  # (f(s_t) - a_t)
            loss *= incremental_reward
            loss.backward(retain_graph=True)
            if cfg.backprop_through_input:
                if hidden_states[0] is None:
                    break
                if hidden_states[0].grad is not None:  # can't clear gradient if it hasn't been back propagated once through
                    hidden_states[0].grad.data.zero_()
                curr_grad = hidden_states[0].grad
                torch.nn.utils.clip_grad_value_(curr_grad, cfg.clip_value)
                hidden_states[1].backward(curr_grad, retain_graph=True)
        assert self.rewards == []

    def should_update_weights(self, reward, is_episode_done):
        if np.abs(reward) > cfg.reward_update_min and is_episode_done:
            return True
        return False

    def calc_aux_reward(self):
        self.aux_reward = 0
        if self.is_life_env:
            if self.model.get_env_pred_val() is not None:  # todo
                self.aux_reward += cfg.life.EXPLOITATION_PENALTY
            if cfg.self_reward_update:
                self.aux_reward += cfg.reward_prediction_discount * cfg.adjacent_reward_list[self.pred_feel_val.value]
        return self.aux_reward

    def log_predictions(self, writer=sys.stdout):
        writer.write('\nAgent Summary at timestep ' + str(self.t) + '\n')
        writer.write('prediction to environment: ' + str(self.model.get_env_pred_val()) + '\n')
        writer.write(str(self.pred_val) + ', ' + str(self.pred_feel_val))
        writer.write('\n\n full reward is: ' + str(self.reward))
        writer.write('\n')
        writer.flush()

    def store_results(self, reward):
        self.testing_rewards.insert(self.t, reward)
        if not self.t % cfg.rewards_eval_window:
            self.mean_testing_rewards.append(np.average(self.testing_rewards.getData()))

    def plot_results(self):
        if cfg.results_path is not None:
            print(self.mean_testing_rewards)
            print(cfg.results_path)
            plt.plot(self.mean_testing_rewards)
            plt.savefig(cfg.results_path + 'averageRewards.png')
