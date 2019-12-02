import logging
import sys

import numpy as np
import torch
import torch.nn.functional as F

import settings
import utils.model_utils as model_utils
from agent_algorithms.factory import register_agent


def logLoss(output, target):
    loss = torch.sum(torch.log(output))
    return loss


@register_agent
class CRAAgent():
    # this agent can work with environments x, y, z (life and gym envs)
    # todo move cragent controller here, and move this stuff in life network
    # try to make the encoding part separate
    def __init__(self, is_episodic, cfg, actor, critic=None):
        self.shared_parameters = critic is None
        self.ac = None
        self.cfg = cfg
        if self.shared_parameters:
            self.ac = actor
            self.n_actions = self.ac.n_actions
            self.learnable_variance = [torch.nn.Parameter(torch.randn(1)), torch.nn.Parameter(torch.randn(1))]
            self.ac.add_parameters(self.learnable_variance)
        else:
            self.actor = actor
            self.n_actions = self.actor.n_actions
            self.critic = critic

        self.pred_val, self.pred_feel_val = None, None
        self.reward = 0

        self.outputs = []
        self.rewards = []
        self.value_estimates = []
        self.aux_estimates = []
        self.action_probs = []
        self.action_taken_probs = []

        self.t = 0

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.update_threshold = cfg.get('update_threshold', -1)
        self.td_step = cfg.get('td_step', -1)
        self.discount_factor = cfg.get('discount_factor', settings.defaults.DISCOUNT_FACTOR)
        self.entropy_coef = cfg.get('entropy_coef', settings.defaults.ENTROPY_COEF)
        self.entropy_coef = cfg.get('entropy_coef', settings.defaults.ENTROPY_COEF)

        logging.debug(' update_threshold : ', self.update_threshold)
        logging.debug(' td_step : ', self.td_step)
        logging.debug(' discount_factor : ', self.discount_factor, '\n')
        logging.debug(' entropy_coef : ', self.entropy_coef, '\n')

    def step(self, env_input):
        env_input = model_utils.convert_env_input(env_input)
        if self.ac is not None:
            action_probs, critic_estimates, aux_estimates = self.ac.forward(env_input)
        else:
            action_probs = self.actor.forward(env_input)
            critic_estimates, aux_estimates = self.critic.forward(env_input)

        self.action_probs.append(action_probs.squeeze(0))
        self.value_estimates.append(critic_estimates.squeeze(0))
        self.aux_estimates.append(aux_estimates.squeeze(0))

        action = np.random.choice(self.n_actions, p=self.action_probs[-1].detach().cpu().numpy())
        self.action_taken_probs.append(self.action_probs[-1][action])

        self.t += 1
        return action

    def update_policy(self, env_reward, episode_end=True, new_state=None):
        ret_loss = 0
        self.rewards.append(env_reward)
        latest_reward = env_reward  # + self.aux_rewards[-1]
        should_update = self.should_update(episode_end, latest_reward)
        if should_update:
            V_target = [0]
            # rewards = torch.tensor(self.rewards).to(settings.DEVICE)
            # aux_rewards = torch.tensor(self.aux_rewards).to(settings.DEVICE)
            while self.rewards:
                # latest reward + (future reward * gamma)
                reward = self.rewards.pop(-1)  # + self.aux_rewards.pop(-1)
                V_target.insert(0, reward + (self.discount_factor * V_target[0]))
                # discounted_rewards.insert(0, reward)
            V_target.pop(-1)  # remove the extra 0 placed before the loop

            V_target = torch.tensor(V_target).to(settings.DEVICE)
            V_estimate = torch.cat(self.value_estimates, dim=0)

            advantage = V_target - V_estimate
            if V_target.shape[0] > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)  # normalizing the advantage

            action_prob_vector = torch.stack(self.action_probs)
            taken_action_probs_vector = torch.stack(self.action_taken_probs)

            action_log_prob = torch.log(taken_action_probs_vector)

            actor_loss = (.5 / (1e-5+self.learnable_variance[0] ** 2.)) * (-action_log_prob * advantage.detach()).sum() + \
                         torch.log(self.learnable_variance[0] + 1e-5)

            entropy_loss = (torch.log(action_prob_vector) * action_prob_vector).sum()

            critic_loss = (1. / (1e-5+self.learnable_variance[1] ** 2.)) * F.smooth_l1_loss(input=V_estimate,
                                                                                       target=V_target,
                                                                                       reduction='sum') + \
                          torch.log(self.learnable_variance[1] + 1e-5)  # .5 * advantage.pow(2).mean()

            loss = actor_loss + critic_loss + (self.entropy_coef * entropy_loss)
            loss.backward()
            ret_loss = loss.detach().cpu().item()
            self.update_networks()
            self.reset_buffers()
        return ret_loss

    def update_networks(self):
        if self.shared_parameters:
            self.ac.update_parameters()
        else:
            self.actor.update_parameters()
            self.critic.update_parameters()

    def should_update(self, episode_end, reward):
        steps_since_update = len(self.rewards) + 1
        td_update = self.td_step != -1 and steps_since_update % self.td_step == 0
        if self.update_threshold == -1:  # not trying the threshold updater
            return episode_end or td_update
        return episode_end or reward >= self.update_threshold

    def reset_buffers(self):
        self.outputs = []
        self.rewards = []
        self.value_estimates = []
        self.aux_estimates = []
        self.action_probs = []
        self.action_taken_probs = []

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
