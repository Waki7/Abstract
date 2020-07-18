from typing import Union

import numpy as np

import utils.experiment_utils as exp_utils
import utils.model_utils as model_utils
from agent_algorithms.factory import register_agent
from networks.basic_fc_networks import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
type = torch.float
args = {'device': device, 'dtype': type}


@register_agent
class A2CAgent():
    # this agent can work with environments x, y, z (life and gym envs)
    # try to make the encoding part separate
    def __init__(self, is_episodic, cfg, actor, critic=None):
        self.is_ac_shared = critic is None
        if self.is_ac_shared:
            self.ac: BaseFCNetwork = actor
            self.n_actions = self.ac.n_actions
        else:
            self.actor: BaseFCNetwork = actor
            self.critic: BaseFCNetwork = critic
            self.n_actions = self.actor.n_actions

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.cfg = {}
        self.update_threshold = exp_utils.copy_config_param(cfg, self.cfg, 'update_threshold', -1)
        self.td_step = exp_utils.copy_config_param(cfg, self.cfg, 'td_step', -1)
        self.discount_factor = exp_utils.copy_config_param(cfg, self.cfg, 'discount_factor',
                                                           settings.defaults.DISCOUNT_FACTOR)
        self.entropy_coef = exp_utils.copy_config_param(cfg, self.cfg, 'entropy_coef', settings.defaults.ENTROPY_COEF)

        self.is_episodic = is_episodic
        self.reward = 0

        self.batch_actions = []
        self.batch_probs_selected = []
        self.batch_probs = []
        self.batch_value_estimates = []
        self.batch_rewards = []
        self.batch_episode_ends = []
        self.t = 0

    def get_action(self):
        if len(self.batch_actions) == 0:
            return None
        return self.batch_actions[-1]

    def step(self, env_input: Union[List[torch.Tensor], torch.Tensor]):
        if self.is_ac_shared:
            probs, estimates = self.ac.forward(env_input)
        else:
            probs = self.actor.forward(env_input)
            estimates = self.critic.forward(env_input)
        try:
            batch_actions = model_utils.random_choice_prob_batch(self.n_actions,
                                                                 probs.detach().cpu().numpy())
        except ValueError as e:
            logging.error('probs are {}'.format(probs))
            logging.error('values in input are {}'.format(np.unique(env_input[0].cpu().numpy())))
            raise e
        selected_probs = torch.stack([probs[i][action] for i, action in enumerate(batch_actions)])

        self.batch_actions.append(batch_actions)
        self.batch_probs_selected.append(selected_probs)
        self.batch_probs.append(probs)
        self.batch_value_estimates.append(estimates.squeeze(-1))

        self.t += 1
        return self.batch_actions[-1]

    def update_policy(self, batch_reward: torch.tensor, batch_episode_end: torch.tensor, **kwargs):
        ret_loss = {}
        self.batch_rewards.append(batch_reward)
        self.batch_episode_ends.append(batch_episode_end)
        should_update = self.batch_should_update(batch_episode_end, batch_reward)

        if should_update:
            reward_vec = torch.stack(self.batch_rewards)
            is_done_vec = torch.stack(self.batch_episode_ends)

            value_estimate_vec = torch.stack(self.batch_value_estimates)
            probs_vec = torch.stack(self.batch_probs)
            selected_prob_vec = torch.stack(self.batch_probs_selected)
            discounted_rewards_vec = model_utils.discount_rewards(rewards=reward_vec, discount=self.discount_factor,
                                                                  td_step=self.td_step)

            advantage = discounted_rewards_vec - value_estimate_vec
            if advantage.shape[0] > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1.e-4)  # normalizing the advantage

            action_log_prob = torch.log(selected_prob_vec)

            zero_done_mask = torch.bitwise_not(is_done_vec).to(settings.DTYPE_X)

            actor_loss = ((-action_log_prob * advantage.detach()) * zero_done_mask).mean()
            critic_loss = (F.smooth_l1_loss(input=value_estimate_vec, target=discounted_rewards_vec,
                                            reduction='none') * zero_done_mask).mean()  # .5 * advantage.pow(2).mean()

            entropy_loss = ((torch.log(probs_vec) * probs_vec).sum(dim=-1) * zero_done_mask).mean()

            ac_loss = actor_loss + critic_loss + (self.entropy_coef * entropy_loss)

            ac_loss.backward()
            ret_loss['actor_loss'] = actor_loss.detach().cpu().item()
            ret_loss['critic_loss'] = critic_loss.detach().cpu().item()
            self.update_networks()
            self.reset_buffers()
        return ret_loss

    def update_networks(self):
        if self.is_ac_shared:
            self.ac.update_parameters()
        else:
            self.actor.update_parameters()
            self.critic.update_parameters()

    def reset_buffers(self):
        self.batch_actions = []
        self.batch_probs_selected = []
        self.batch_probs = []
        self.batch_value_estimates = []
        self.batch_rewards = []
        self.batch_episode_ends = []
        if self.is_ac_shared:
            self.ac.reset_time()
        else:
            self.actor.reset_time()
            self.critic.reset_time()

    def batch_should_update(self, batch_episode_end, batch_reward):
        steps_since_update = len(self.batch_rewards) + 1
        td_update = self.td_step != -1 and steps_since_update % self.td_step == 0
        if self.update_threshold == -1:  # not trying the threshold updater
            return all(batch_episode_end) or td_update
        update = batch_episode_end or np.abs(batch_reward) >= self.update_threshold
        return update

    def save(self, path):
        # save parameters
        # save network weights
        # save pickle s
        pass
