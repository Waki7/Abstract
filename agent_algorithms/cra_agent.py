from agent_algorithms.factory import register_agent
import numpy as np
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import gym_life.envs.life_env as life_env
import logging
from utils.TimeBuffer import TimeBuffer
from tensorboardX import SummaryWriter
import settings


def logLoss(output, target):
    loss = torch.sum(torch.log(output))
    return loss

@register_agent
class CRAgent():
    # this agent can work with environments x, y, z (life and gym envs)
    # todo move cragent controller here, and move this stuff in life network
    # try to make the encoding part separate
    def __init__(self, actor, critic, is_episodic, cfg):
        self.actor = actor
        self.critic = critic

        self.pred_val, self.pred_feel_val = None, None
        self.reward = 0

        self.initial_state = torch.zeros((1, self.model.hidden_in_size), **settings.ARGS)
        self.hidden_states = [(None, self.initial_state)]
        self.outputs = []
        self.rewards = []
        self.actions = []

        self.testing_rewards = TimeBuffer(cfg.rewards_eval_window)
        self.mean_testing_rewards = []
        self.t = 0

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.update_threshold = cfg.get('update_threshold', -1)
        self.td_step = cfg.get('td_step', -1)
        self.discount_factor = cfg.get('discount_factor', settings.defaults.DISCOUNT_FACTOR)
        self.entropy_coef = cfg.get('entropy_coef', settings.defaults.ENTROPY_COEF)
        logging.debug(' update_threshold : ', self.update_threshold)
        logging.debug(' td_step : ', self.td_step)
        logging.debug(' discount_factor : ', self.discount_factor, '\n')
        logging.debug(' entropy_coef : ', self.entropy_coef, '\n')

        getattr(torch.optim, cfg.life.OPTIMIZER)(self.model.parameters(), lr=cfg.life.LR)
        self.criterion = nn.MSELoss()

        self.is_life_env = isinstance(self.env, life_env.LifeEnv)

        self.writer = SummaryWriter()

    def step(self, env_input):
        action, self.aux_reward = self.forward(env_input)
        self.actions.append(action)
        self.t += 1
        return action

    def update_policy(self, env_reward, episode_end=True):
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
        action = torch.argmax(output) # WHERE THE FUCK DO WE STORE THIS WHOLE SPECIFIC TO THE ENVIRONMENT BULLSHIT
        # action = self.model.get_action_vector(output)
        self.outputs.append(output)
        self.hidden_states.append((hidden_input, hidden_output))
        return action

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

    def should_update_weights(self, reward, is_episode_done):
        if np.abs(reward) > cfg.reward_update_min and is_episode_done:
            return True
        return False

    def calc_aux_reward(self):
        return self.aux_reward

    def get_focus(self):
        return self.pred_val  # need to make a difference if to self or to environment

    def get_reward_perception(self):
        return self.pred_feel_val

    def get_env_pred_val(self):
        return self.pred_val if self.pred < self.out_vector_idx[len(self.action_channels)] else None

    def get_action(self):
        return self.get_env_pred_val()  # todo we want to implement this in the agent i think...

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
