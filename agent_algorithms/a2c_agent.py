import logging
from agent_algorithms.factory import register_agent
import sys
import torch.nn.functional as F
from utils.model_utils import true_with_probability
import matplotlib.pyplot as plt
from utils.TimeBuffer import TimeBuffer
from networks.base_networks import *
from tensorboardX import SummaryWriter

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
        self.ac = None
        if self.is_ac_shared:
            self.ac = actor
            self.n_actions = self.ac.n_actions
        else:
            self.actor = actor
            self.n_actions = self.actor.n_actions
            self.critic = critic

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

        self.is_episodic = is_episodic
        self.reward = 0
        self.action_probs = []
        self.probs = []
        self.rewards = []
        self.value_estimates = []
        self.t = 0

    def step(self, env_input):
        env_input = torch.from_numpy(env_input).to(settings.DEVICE).float().unsqueeze(0)
        if self.ac is not None:
            probs, estimates = self.ac.forward(env_input)
        else:
            probs = self.actor.forward(env_input).squeeze(0)
            estimates = self.critic.forward(env_input).squeeze(0)

        self.probs.append(probs)
        self.value_estimates.append(estimates)

        action = np.random.choice(self.n_actions, p=probs.detach().cpu().numpy())
        self.action_probs.append(probs[action])

        self.t += 1
        return action

    def update_policy(self, env_reward, episode_end, new_state=None):
        ret_loss = 0
        self.rewards.append(env_reward)
        should_update = self.should_update(episode_end, env_reward)
        if should_update:
            discounted_rewards = [0]
            while self.rewards:
                # latest reward + (future reward * gamma)
                discounted_rewards.insert(0, self.rewards.pop(-1) + (self.discount_factor * discounted_rewards[0]))
            discounted_rewards.pop(-1)  # remove the extra 0 placed before the loop

            Q_val = torch.tensor(discounted_rewards).to(**args)
            V_estimate = torch.cat(self.value_estimates, dim=0)
            advantage = Q_val - V_estimate
            if len(discounted_rewards) > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)  # normalizing the advantage

            probs = torch.stack(self.probs)
            action_probs = torch.stack(self.action_probs)

            action_log_prob = torch.log(action_probs)
            actor_loss = (-action_log_prob * advantage.detach()).sum()

            critic_loss = F.smooth_l1_loss(input=V_estimate, target=Q_val,
                                           reduction='sum')  # .5 * advantage.pow(2).mean()

            entropy_loss = (torch.log(probs) * probs).sum()

            ac_loss = actor_loss + critic_loss + (self.entropy_coef * entropy_loss)

            ac_loss.backward()
            ret_loss = ac_loss.detach().cpu().item()

            self.update_networks()

        if should_update:
            self.reset_buffers()
        return ret_loss

    def update_networks(self):
        if self.is_ac_shared:
            self.ac.update_parameters()
        else:
            self.actor.update_parameters()
            self.critic.update_parameters()

    def reset_buffers(self):
        self.rewards = []
        self.probs = []
        self.action_probs = []
        self.value_estimates = []

    def should_update(self, episode_end, reward):
        steps_since_update = len(self.rewards) + 1
        td_update = self.td_step != -1 and steps_since_update % self.td_step == 0
        if self.update_threshold == -1:  # not trying the threshold updater
            return episode_end or td_update
        return episode_end or reward >= self.update_threshold
