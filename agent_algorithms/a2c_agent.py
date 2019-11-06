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
    def __init__(self, actor, critic, is_episodic, cfg):
        self.actor = actor
        self.critic = critic

        self.is_episodic = is_episodic
        self.n_step = cfg.get('n_step', -1)
        self.update_threshold = cfg.get('update_threshold', -1)
        self.random_update_prob = .1

        self.reward = 0
        self.average_rewards = []
        self.log_probs = []
        self.action_probs = []
        self.probs = []
        self.rewards = []
        self.value_estimates = []
        self.t = 0
        self.continuous_episode_length = cfg.pg.CONTINUOUS_EPISODE_LENGTH
        self.optimizer = getattr(torch.optim, cfg.gym.OPTIMIZER)(self.model.parameters(), lr=cfg.gym.LR)
        self.writer = SummaryWriter()

    def step(self, env_input):
        action, log_prob, value_estimate = self.model.get_action(env_input)
        self.value_estimates.append(value_estimate)
        self.log_probs.append(log_prob)
        self.t += 1
        return action

    def update_policy(self, env_reward, episode_end):
        self.rewards.append(env_reward)

        if self.should_update(episode_end, env_reward):
            discounted_rewards = [0]
            while self.rewards:
                # latest reward + (future reward * gamma)
                discounted_rewards.insert(0, self.rewards.pop() + (cfg.discount_factor * discounted_rewards[0]))
            discounted_rewards.pop(-1)  # remove the extra 0 placed before the loop

            Q_val = torch.tensor(discounted_rewards).to(**args)
            Q_val = (Q_val - Q_val.mean()) / (Q_val.std() + 1e-9) # normalizing the advantage
            V_estimate = torch.tensor(self.value_estimates).to(**args)
            advantage = Q_val - V_estimate
            log_prob = torch.stack(self.log_probs)

            actor_loss = (-log_prob * advantage.detach()).mean()
            critic_loss = F.smooth_l1_loss(input=V_estimate, target=Q_val)#.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            ac_loss.backward()

            self.optimizer.step()

            self.t = 0
            self.rewards = []
            self.log_probs = []


    def should_update(self, episode_end, reward):
        if self.is_episodic:
            return episode_end
        if self.n_step == -1:
            return (self.t + 1) % self.n_step == 0
        if self.update_threshold == -1:
            return true_with_probability(self.random_update_prob)
        return reward >= self.update_threshold
