from agent_algorithms.factory import register_agent
import numpy as np
import torch
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

        self.pred_val, self.pred_feel_val = None, None
        self.reward = 0

        self.outputs = []
        self.rewards = []
        self.aux_rewards = []
        self.action_probs = []

        self.t = 0

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.update_threshold = cfg.get('update_threshold', -1)
        self.td_step = cfg.get('td_step', -1)
        self.discount_factor = cfg.get('discount_factor', settings.defaults.DISCOUNT_FACTOR)
        self.entropy_coef = cfg.get('entropy_coef', settings.defaults.ENTROPY_COEF)
        self.reward_update_min = cfg.get('reward_update_min', 0.0)
        logging.debug(' update_threshold : ', self.update_threshold)
        logging.debug(' td_step : ', self.td_step)
        logging.debug(' discount_factor : ', self.discount_factor, '\n')
        logging.debug(' entropy_coef : ', self.entropy_coef, '\n')
        logging.debug(' reward_update_min : ', self.reward_update_min, '\n')

        self.criterion = nn.MSELoss()

    def step(self, env_input):
        env_input = model_utils.convert_env_input(env_input)
        action, aux_reward = self.actor.forward(env_input)
        self.action_probs.append(action)
        self.aux_rewards.append(aux_reward)
        action_taken = torch.argmax(action, dim=-1)
        self.t += 1
        return action_taken.item()

    def update_policy(self, env_reward, episode_end=True, new_state=None):
        self.rewards.append(env_reward)
        latest_reward = env_reward + self.aux_rewards[-1]
        should_update = self.should_update_weights(latest_reward, episode_end)
        if should_update:
            discounted_rewards = [0]
            while self.rewards:
                # latest reward + (future reward * gamma)
                reward = self.rewards.pop(-1) + self.aux_rewards.pop(-1)
                discounted_rewards.insert(0, reward + (self.discount_factor * discounted_rewards[0]))
            discounted_rewards.pop(-1)  # remove the extra 0 placed before the loop

            reward_vector = torch.tensor(discounted_rewards).to(settings.DEVICE)

            action_prob_vector = torch.stack(self.action_probs)
            one_hot = model_utils.one_hot(action_prob_vector)


            loss = self.criterion(input=action_prob_vector, target=one_hot)
            loss = reward_vector * loss
            loss.backward()

            self.actor.update_parameters()
            self.reset_buffers()

    def should_update_weights(self, reward, is_episode_done):
        if torch.abs(reward) > self.reward_update_min or is_episode_done:
            return True
        return False

    def reset_buffers(self):
        self.outputs = []
        self.rewards = []
        self.aux_rewards = []
        self.action_probs = []

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
