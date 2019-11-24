from agent_algorithms.factory import register_agent
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import utils.model_utils as model_utils
import logging
import settings


def logLoss(output, target):
    loss = torch.sum(torch.log(output))
    return loss


@register_agent
class CRAAgent():
    # this agent can work with environments x, y, z (life and gym envs)
    # todo move cragent controller here, and move this stuff in life network
    # try to make the encoding part separate
    def __init__(self, is_episodic, actor, cfg):
        self.actor = actor
        self.n_actions = self.actor.n_actions

        self.pred_val, self.pred_feel_val = None, None
        self.reward = 0

        self.outputs = []
        self.rewards = []
        self.aux_rewards = []
        self.action_probs = []
        self.actions_taken = []
        self.action_taken_probs = []

        self.t = 0

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.update_threshold = cfg.get('update_threshold', -1)
        self.td_step = cfg.get('td_step', -1)
        self.discount_factor = cfg.get('discount_factor', settings.defaults.DISCOUNT_FACTOR)
        self.entropy_coef = cfg.get('entropy_coef', settings.defaults.ENTROPY_COEF)
        self.supervised_loss = cfg.get('supervised_loss', False)
        logging.debug(' update_threshold : ', self.update_threshold)
        logging.debug(' td_step : ', self.td_step)
        logging.debug(' discount_factor : ', self.discount_factor, '\n')
        logging.debug(' entropy_coef : ', self.entropy_coef, '\n')
        logging.debug(' supervised_loss : ', self.supervised_loss, '\n')

    def step(self, env_input):
        env_input = model_utils.convert_env_input(env_input)
        action_probs, aux_reward = self.actor.forward(env_input)
        action_probs = action_probs.squeeze(0)
        # aux_reward = aux_reward.squeeze(0)
        # self.aux_rewards.append(aux_reward)
        action = np.random.choice(self.n_actions, p=action_probs.detach().cpu().numpy())
        self.action_probs.append(action_probs)
        self.actions_taken.append(action)
        self.action_taken_probs.append(action_probs[action])
        self.t += 1
        return action

    def update_policy(self, env_reward, episode_end=True, new_state=None):
        self.rewards.append(env_reward)
        latest_reward = env_reward #+ self.aux_rewards[-1]
        should_update = self.should_update(episode_end, latest_reward)
        if should_update:
            discounted_rewards = [0]
            # rewards = torch.tensor(self.rewards).to(settings.DEVICE)
            # aux_rewards = torch.tensor(self.aux_rewards).to(settings.DEVICE)

            while self.rewards:
                # latest reward + (future reward * gamma)
                reward = self.rewards.pop(-1) #+ self.aux_rewards.pop(-1)
                discounted_rewards.insert(0, reward + (self.discount_factor * discounted_rewards[0]))
                # discounted_rewards.insert(0, reward)
            discounted_rewards.pop(-1)  # remove the extra 0 placed before the loop

            discounted_rewards = torch.tensor(discounted_rewards).to(settings.DEVICE)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

            if self.supervised_loss:
                action_prob_vector = torch.stack(self.action_probs)
                idx_vector = torch.tensor(self.actions_taken).to(settings.DEVICE).unsqueeze(-1)
                one_hot = model_utils.one_hot(action_prob_vector,
                                              idx=idx_vector)
                # loss = F.smooth_l1_loss(input=action_prob_vector, target=one_hot)
                loss = (action_prob_vector - one_hot).pow(2).sum(dim=-1)
                loss = (discounted_rewards * loss).sum()
                # loss += F.smooth_l1_loss(input=aux_rewards, target=discounted_rewards)
            else:
                taken_action_prob_vector = torch.stack(self.action_taken_probs)
                action_log_prob = torch.log(taken_action_prob_vector)
                loss = (-action_log_prob * discounted_rewards).sum()

            loss.backward()

            self.actor.update_parameters()
            self.reset_buffers()

    def should_update(self, episode_end, reward):
        steps_since_update = len(self.rewards) + 1
        td_update = self.td_step != -1 and steps_since_update % self.td_step == 0
        if self.update_threshold == -1:  # not trying the threshold updater
            return episode_end or td_update
        return episode_end or reward >= self.update_threshold

    def reset_buffers(self):
        self.outputs = []
        self.rewards = []
        self.aux_rewards = []
        self.action_probs = []
        self.actions_taken = []
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
